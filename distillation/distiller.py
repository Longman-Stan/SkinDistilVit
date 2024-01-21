from transformers import PreTrainedModel, PretrainedConfig
from .distiller_config import DistillerConfig
from .distillation_output import DistillerOutput
from collections import OrderedDict
import warnings
import torch.nn as nn
import torch
import time

class Distiller(PreTrainedModel):

    def __init__(self, config: DistillerConfig, student: PreTrainedModel, teacher: PreTrainedModel):
        super().__init__(config)
        self.distiller_config = config
        self.student = student
        self.config = student.config
        self.num_labels = self.student.num_labels
        self.teacher = teacher
        for param in self.teacher.parameters():
            param.requires_grad = False
        self.teacher.eval()
        self.student.train()

        self.ce_loss_fct = nn.KLDivLoss(reduction="batchmean")
        self.lm_loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
    
        if self.distiller_config.alpha_mse > 0.0:
            self.mse_loss_fct = nn.MSELoss(reduction="sum")
        if self.distiller_config.alpha_cos > 0.0:
            self.cosine_loss_fct = nn.CosineEmbeddingLoss(reduction="mean")

    
    def state_dict(self, *args, destination=None, prefix='', keep_vars=False):
        
        if len(args) > 0:
            if destination is None:
                destination = args[0]
            if len(args) > 1 and prefix == '':
                prefix = args[1]
            if len(args) > 2 and keep_vars is False:
                keep_vars = args[2]
            # DeprecationWarning is ignored by default
            warnings.warn(
                "Positional args are being deprecated, use kwargs instead. Refer to "
                "https://pytorch.org/docs/master/generated/torch.nn.Module.html#torch.nn.Module.state_dict"
                " for details.")

        if destination is None:
            destination = OrderedDict()
            destination._metadata = OrderedDict()

        local_metadata = dict(version=self._version)
        if hasattr(destination, "_metadata"):
            destination._metadata[prefix[:-1]] = local_metadata

        self._save_to_state_dict(destination, prefix, keep_vars)
        for name, module in self.student._modules.items():
            if module is not None:
                module.state_dict(destination=destination, prefix=prefix + name + '.', keep_vars=keep_vars)
        for hook in self._state_dict_hooks.values():
            hook_result = hook(self, destination, prefix, local_metadata)
            if hook_result is not None:
                destination = hook_result
        return destination

    def forward(self, **inputs):

        inputs["output_hidden_states"] = True


        #print(self.teacher.device)
        #print(self.student.device)

        #a = time.time()
        with torch.no_grad():
           teacher_pred = self.teacher(**inputs)
        #print("teacher inference", time.time() - a)

        #a = time.time()
        student_pred = self.student(**inputs)
        #print("Student inference", time.time() - a)

        a = time.time()
        if isinstance(teacher_pred, dict):
            t_logits = teacher_pred["logits"]
            t_hidden_states = teacher_pred["hidden_states"]
        else:
            _, t_logits, t_hidden_states = teacher_pred
        

        if isinstance(student_pred, dict):
            s_loss = student_pred["loss"]
            s_logits = student_pred["logits"]
            s_hidden_states = student_pred["hidden_states"]
        else:
            s_loss, s_logits, s_hidden_states = student_pred
            
        #print("Extracting params", time.time() -a)

        #assert s_logits.size() == t_logits.size()

        #a = time.time()
        loss_ce =    self.ce_loss_fct(
                nn.functional.log_softmax(s_logits / self.distiller_config.temperature, dim=-1),
                nn.functional.softmax(t_logits / self.distiller_config.temperature, dim=-1),
            ) * (self.distiller_config.temperature) ** 2
        #print("loss ce", time.time() - a)
        
        loss = self.distiller_config.alpha_ce * loss_ce

        loss += self.distiller_config.alpha_task * s_loss

        if self.distiller_config.alpha_mse > 0.0:
            loss_mse = self.mse_loss_fct(s_logits, t_logits) / s_logits.size(0)  # Reproducing batchmean reduction
            loss += self.distiller_config.alpha_mse * loss_mse
        
        if self.distiller_config.alpha_cos > 0.0:
            s_hidden_states = s_hidden_states[-1]
            t_hidden_states = t_hidden_states[-1]
            #assert s_hidden_states.size() == t_hidden_states.size()
            dim = s_hidden_states.size(-1)

            s_hidden_states = s_hidden_states.view(-1, dim)
            t_hidden_states = t_hidden_states.view(-1, dim)

            target = s_hidden_states.new(s_hidden_states.size(0)).fill_(1)
            loss_cos = self.cosine_loss_fct(s_hidden_states, t_hidden_states, target)
            loss += self.distiller_config.alpha_cos * loss_cos

        return_dict = inputs.get("return_dict",None)
        return_dict = return_dict if return_dict is not None else self.distiller_config.use_return_dict
        if not return_dict:
            return (loss, s_logits, t_logits, s_hidden_states, t_hidden_states)
        
        return DistillerOutput(
            loss = loss,
            s_logits = s_logits,
            t_logits = t_logits,
            s_hidden_states = s_hidden_states, 
            t_hidden_states = t_hidden_states
        )
