from transformers import PretrainedConfig

class DistillerConfig(PretrainedConfig):

    def __init__(self,
        alpha_ce: float = 0.5,
        alpha_task: float = 0.0,
        alpha_mse: float = 0.0,
        alpha_cos: float = 0.0,
        temperature: float = 1.0,
        **kwargs):
        super().__init__(**kwargs)
        self.alpha_task = alpha_task
        self.alpha_ce = alpha_ce
        self.alpha_mse = alpha_mse
        self.alpha_cos = alpha_cos
        self.temperature = temperature
