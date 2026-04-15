import torch
import torch.nn as nn
from torch.distributions.normal import Normal
import torch.nn.functional as F

try:
    from timm.models.vision_transformer import Mlp
except ImportError as exc:  # pragma: no cover - optional dependency guard
    Mlp = None
    TIMM_IMPORT_ERROR = exc
else:
    TIMM_IMPORT_ERROR = None


def require_timm() -> None:
    if TIMM_IMPORT_ERROR is not None:
        raise ImportError(
            "ARROW requires the optional dependency 'timm'. Install it before instantiating the ARROW model."
        ) from TIMM_IMPORT_ERROR


class SparseDispatcher:
    def __init__(self, num_experts, gates):
        self._gates = gates
        self._num_experts = num_experts
        sorted_experts, index_sorted_experts = torch.nonzero(gates).sort(0)
        _, self._expert_index = sorted_experts.split(1, dim=1)
        self._batch_index = torch.nonzero(gates)[index_sorted_experts[:, 1], 0]
        self._part_sizes = (gates > 0).sum(0).tolist()
        gates_exp = gates[self._batch_index.flatten()]
        self._nonzero_gates = torch.gather(gates_exp, 1, self._expert_index)

    def dispatch(self, inp):
        inp_exp = inp[self._batch_index].squeeze(1)
        return torch.split(inp_exp, self._part_sizes, dim=0)

    def combine(self, expert_out, multiply_by_gates=True):
        stitched = torch.cat(expert_out, 0)
        if multiply_by_gates:
            stitched = stitched.mul(self._nonzero_gates)
        zeros = torch.zeros(
            self._gates.size(0),
            expert_out[-1].size(1),
            requires_grad=True,
            device=stitched.device,
        )
        combined = zeros.index_add(0, self._batch_index, stitched.float())
        return combined

    def expert_to_gates(self):
        return torch.split(self._nonzero_gates, self._part_sizes, dim=0)


class MoE(nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
        hidden_size,
        routed_num_experts,
        shared_num_experts=1,
        noisy_gating=True,
        k=2,
    ):
        super().__init__()
        require_timm()
        self.noisy_gating = noisy_gating
        self.routed_num_experts = routed_num_experts
        self.shared_num_experts = shared_num_experts
        self.output_size = output_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.k = k
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.shared_experts = Mlp(
            in_features=input_size,
            hidden_features=self.shared_num_experts * hidden_size,
            out_features=output_size,
            act_layer=approx_gelu,
            drop=0,
        )
        self.routed_experts = nn.ModuleList(
            [
                Mlp(
                    in_features=input_size,
                    hidden_features=hidden_size,
                    act_layer=approx_gelu,
                    drop=0,
                )
                for _ in range(self.routed_num_experts)
            ]
        )
        self.w_gate = nn.Parameter(torch.zeros(input_size, self.routed_num_experts), requires_grad=True)
        self.w_noise = nn.Parameter(torch.zeros(input_size, self.routed_num_experts), requires_grad=True)
        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(1)
        self.register_buffer("mean", torch.tensor([0.0]))
        self.register_buffer("std", torch.tensor([1.0]))
        if self.k > self.routed_num_experts:
            raise ValueError("k must be <= routed_num_experts")

    def cv_squared(self, x):
        eps = 1e-10
        if x.shape[0] == 1:
            return torch.tensor([0], device=x.device, dtype=x.dtype)
        return x.float().var() / (x.float().mean() ** 2 + eps)

    def _gates_to_load(self, gates):
        return (gates > 0).sum(0)

    def _prob_in_top_k(self, clean_values, noisy_values, noise_stddev, noisy_top_values):
        batch = clean_values.size(0)
        m = noisy_top_values.size(1)
        top_values_flat = noisy_top_values.flatten()
        threshold_positions_if_in = torch.arange(batch, device=clean_values.device) * m + self.k
        threshold_if_in = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_in), 1)
        is_in = torch.gt(noisy_values, threshold_if_in)
        threshold_positions_if_out = threshold_positions_if_in - 1
        threshold_if_out = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_out), 1)
        normal = Normal(self.mean, self.std)
        prob_if_in = normal.cdf((clean_values - threshold_if_in) / noise_stddev)
        prob_if_out = normal.cdf((clean_values - threshold_if_out) / noise_stddev)
        return torch.where(is_in, prob_if_in, prob_if_out)

    def noisy_top_k_gating(self, x, train, noise_epsilon=1e-2):
        clean_logits = x @ self.w_gate
        if self.noisy_gating and train:
            raw_noise_stddev = x @ self.w_noise
            noise_stddev = self.softplus(raw_noise_stddev) + noise_epsilon
            noisy_logits = clean_logits + (torch.randn_like(clean_logits) * noise_stddev)
            logits = noisy_logits
        else:
            logits = clean_logits
            noisy_logits = clean_logits
            noise_stddev = None
        top_logits, top_indices = logits.topk(min(self.k + 1, self.routed_num_experts), dim=1)
        top_k_logits = top_logits[:, : self.k]
        top_k_indices = top_indices[:, : self.k]
        top_k_gates = self.softmax(top_k_logits)
        zeros = torch.zeros_like(logits, requires_grad=True)
        gates = zeros.scatter(1, top_k_indices, top_k_gates.to(zeros.dtype))
        if self.noisy_gating and self.k < self.routed_num_experts and train:
            load = self._prob_in_top_k(clean_logits, noisy_logits, noise_stddev, top_logits).sum(0)
        else:
            load = self._gates_to_load(gates)
        return gates, load

    def forward(self, x, time_interval, loss_coef=1e-2):
        del time_interval
        shape = x.shape
        x = x.reshape(-1, self.input_size)
        z = self.shared_experts(x)
        gates, load = self.noisy_top_k_gating(x, self.training)
        importance = gates.sum(0)
        self.aux_loss = (self.cv_squared(importance) + self.cv_squared(load)) * loss_coef
        dispatcher = SparseDispatcher(self.routed_num_experts, gates)
        expert_inputs = dispatcher.dispatch(x)
        expert_outputs = [self.routed_experts[i](expert_inputs[i]) for i in range(self.routed_num_experts)]
        y = dispatcher.combine(expert_outputs)
        return (y + z).reshape(*shape)


class Gate(nn.Module):
    def __init__(self, dim, topk, num_experts, num_time_interval=3):
        super().__init__()
        self.dim = dim
        self.k = topk
        self.num_experts = num_experts
        self.num_time_interval = num_time_interval
        self.w_gate_noise = nn.Linear(dim, (self.num_time_interval + 1) * num_experts, bias=False)
        self.softmax = nn.Softmax(dim=-1)
        self.register_buffer("batch_indices", torch.arange(4))

    def forward(self, x, time_interval):
        batch_size, seq_len, _ = x.shape
        scores_noises = F.sigmoid(self.w_gate_noise(x))
        original_scores = scores_noises[:, :, : self.num_experts]
        noises = scores_noises[:, :, self.num_experts :]
        if batch_size != self.batch_indices.shape[0] or self.batch_indices.device != x.device:
            self.batch_indices = torch.arange(batch_size, device=x.device)
        noises = noises.reshape(batch_size, seq_len, self.num_time_interval, self.num_experts)
        noises = noises[self.batch_indices, :, time_interval, :]
        noises_dist = []
        for interval_index in range(self.num_time_interval):
            if (time_interval == interval_index).any():
                noise_dist = torch.mean(noises[time_interval == interval_index], dim=(0, 1))
            else:
                noise_dist = torch.zeros(self.num_experts, device=x.device)
            noises_dist.append(noise_dist)
        noises_dist = torch.stack(noises_dist, dim=0)
        select_scores = (original_scores + noises).reshape(-1, self.num_experts)
        original_scores = original_scores.reshape(-1, self.num_experts)
        top_k_indices = select_scores.topk(min(self.k, self.num_experts), dim=1)[1]
        top_k_gates = self.softmax(original_scores.gather(1, top_k_indices))
        zeros = torch.zeros_like(original_scores, requires_grad=True)
        gates = zeros.scatter(1, top_k_indices, top_k_gates.to(zeros.dtype))
        return gates, noises_dist


class SP_MOE(nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
        hidden_size,
        routed_num_experts,
        shared_num_experts=1,
        noisy_gating=True,
        k=2,
    ):
        del noisy_gating
        super().__init__()
        require_timm()
        self.routed_num_experts = routed_num_experts
        self.shared_num_experts = shared_num_experts
        self.output_size = output_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.k = k
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.shared_experts = Mlp(
            in_features=input_size,
            hidden_features=self.shared_num_experts * hidden_size,
            out_features=output_size,
            act_layer=approx_gelu,
            drop=0,
        )
        self.routed_experts = nn.ModuleList(
            [
                Mlp(
                    in_features=input_size,
                    hidden_features=hidden_size,
                    act_layer=approx_gelu,
                    drop=0,
                )
                for _ in range(self.routed_num_experts)
            ]
        )
        self.gate = Gate(dim=input_size, topk=k, num_experts=routed_num_experts, num_time_interval=3)
        self.aux_loss_dict = {"noises_dist": None}
        if self.k > self.routed_num_experts:
            raise ValueError("k must be <= routed_num_experts")

    def forward(self, x, time_interval):
        shape = x.shape
        gates, noises_dist = self.gate(x, time_interval)
        self.aux_loss_dict["noises_dist"] = noises_dist
        x = x.reshape(-1, self.input_size)
        z = self.shared_experts(x)
        dispatcher = SparseDispatcher(self.routed_num_experts, gates)
        expert_inputs = dispatcher.dispatch(x)
        expert_outputs = [self.routed_experts[i](expert_inputs[i]) for i in range(self.routed_num_experts)]
        y = dispatcher.combine(expert_outputs)
        return (y + z).reshape(*shape)
