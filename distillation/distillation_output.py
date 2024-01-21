from transformers.utils import ModelOutput
from dataclasses import dataclass
from typing import Optional, Tuple
import torch

@dataclass
class DistillerOutput(ModelOutput):
    """
    Base class for outputs of image classification models.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Classification (or regression if config.num_labels==1) loss.
        logits (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`):
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each stage) of shape `(batch_size, sequence_length, hidden_size)`. Hidden-states
            (also called feature maps) of the model at the output of each stage.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, patch_size,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    loss: Optional[torch.FloatTensor] = None
    s_logits: torch.FloatTensor = None
    t_logits: torch.FloatTensor = None
    s_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    t_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    s_attentions: Optional[Tuple[torch.FloatTensor]] = None
    t_attentions: Optional[Tuple[torch.FloatTensor]] = None