from easytsf.runner.exp_base_runner import LTSFRunner

class UnivariateRunner(LTSFRunner):
    def __init__(self, **kwargs):
        super(UnivariateRunner, self).__init__(**kwargs)

    def forward(self, batch, batch_idx):
        var_x, marker_x, var_y, marker_y = [_.float() for _ in batch]
        var_x = var_x[..., self.hparams.var_id:self.hparams.var_id+1]
        label = var_y[:, -self.hparams.pred_len:, self.hparams.var_id:self.hparams.var_id+1]
        prediction = self.model(var_x, marker_x)
        return prediction, label