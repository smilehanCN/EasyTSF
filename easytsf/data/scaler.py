import numpy as np
import torch


class StandardScaler:
    def __init__(self, mean, std):
        self.mean = None
        self.std = None
        self.set_stats(mean, std)

    @classmethod
    def fit(cls, data):
        tensor = torch.as_tensor(data, dtype=torch.float32)

        mean = tensor.mean(dim=0, keepdim=True)
        std = tensor.std(dim=0, keepdim=True, unbiased=False)
        std = torch.where(std == 0, torch.ones_like(std), std)
        return cls(mean, std)

    def set_stats(self, mean, std):
        self.mean = torch.as_tensor(mean, dtype=torch.float32).detach()
        self.std = torch.as_tensor(std, dtype=torch.float32).detach()
        return self

    def transform(self, input_data, mask=None):
        output = (input_data - self.mean) / self.std
        if mask is None:
            return output
        return torch.where(mask, output, input_data)

    def inverse_transform(self, input_data, mask=None):
        output = input_data * self.std + self.mean
        if mask is None:
            return output
        return torch.where(mask, output, input_data)
