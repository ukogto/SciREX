"""
A ``TokenEmbedder`` which uses one of the LONGFORMER models
(https://github.com/google-research/longformer)
to produce embeddings.

At its core it uses Hugging Face's PyTorch implementation
(https://github.com/huggingface/pytorch-pretrained-LONGFORMER),
so thanks to them!
"""
from typing import Dict, List
import logging

import torch
import torch.nn.functional as F

# from pytorch_pretrained_longformer.modeling import LongformerModel

from allennlp.modules.scalar_mix import ScalarMix
from allennlp.modules.token_embedders.token_embedder import TokenEmbedder
from allennlp.nn import util

logger = logging.getLogger(__name__)
from .longformer.longformer.modeling_longformer import LongformerEmbeddings, LongformerModel
class PretrainedLongformerModel:
    """
    In some instances you may want to load the same LONGFORMER model twice
    (e.g. to use as a token embedder and also as a pooling layer).
    This factory provides a cache so that you don't actually have to load the model twice.
    """
    _cache: Dict[str, LongformerModel] = {}

    @classmethod
    def load(cls, model_name: str, cache_model: bool = True) -> LongformerModel:
        if model_name in cls._cache:
            return PretrainedLongformerModel._cache[model_name]

        model = LongformerModel.from_pretrained(model_name)
        if cache_model:
            cls._cache[model_name] = model

        return model


class LongformerEmbedder(TokenEmbedder):
    """
    A ``TokenEmbedder`` that produces LONGFORMER embeddings for your tokens.
    Should be paired with a ``LongformerIndexer``, which produces wordpiece ids.

    Most likely you probably want to use ``PretrainedLongformerEmbedder``
    for one of the named pretrained models, not this base class.

    Parameters
    ----------
    longformer_model: ``LongformerModel``
        The LONGFORMER model being wrapped.
    top_layer_only: ``bool``, optional (default = ``False``)
        If ``True``, then only return the top layer instead of apply the scalar mix.
    max_pieces : int, optional (default: 512)
        The LONGFORMER embedder uses positional embeddings and so has a corresponding
        maximum length for its input ids. Assuming the inputs are windowed
        and padded appropriately by this length, the embedder will split them into a
        large batch, feed them into LONGFORMER, and recombine the output as if it was a
        longer sequence.
    num_start_tokens : int, optional (default: 1)
        The number of starting special tokens input to LONGFORMER (usually 1, i.e., [CLS])
    num_end_tokens : int, optional (default: 1)
        The number of ending tokens input to LONGFORMER (usually 1, i.e., [SEP])
    scalar_mix_parameters: ``List[float]``, optional, (default = None)
        If not ``None``, use these scalar mix parameters to weight the representations
        produced by different layers. These mixing weights are not updated during
        training.
    """
    def __init__(self,
                 longformer_model: LongformerModel,
                 top_layer_only: bool = False,
                 max_pieces: int = 512,
                 num_start_tokens: int = 1,
                 num_end_tokens: int = 1,
                 scalar_mix_parameters: List[float] = None) -> None:
        super().__init__()
        self.longformer_model = longformer_model
        self.output_dim = longformer_model.config.hidden_size
        self.max_pieces = max_pieces
        self.num_start_tokens = num_start_tokens
        self.num_end_tokens = num_end_tokens

        if not top_layer_only:
            self._scalar_mix = ScalarMix(longformer_model.config.num_hidden_layers,
                                         do_layer_norm=False,
                                         initial_scalar_parameters=scalar_mix_parameters,
                                         trainable=scalar_mix_parameters is None)
        else:
            self._scalar_mix = None

    def get_output_dim(self) -> int:
        return self.output_dim

    def forward(self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None, biu_one_hot_encoding=None, bbox=None):
        # biu_one_hot_encoding: int32 tensor of width [batch_size, seq_length,3]
        if position_ids is None:
            if input_ids is not None:
                # Create the position ids from the input token ids. Any padded tokens remain padded.
                position_ids = create_position_ids_from_input_ids(input_ids, self.padding_idx).to(input_ids.device)
            else:
                position_ids = self.create_position_ids_from_inputs_embeds(inputs_embeds)
        
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]

        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.int, device=self.position_ids.device)

        if biu_one_hot_encoding is None:
            print("Embedding.position_ids type:", self.position_ids.device)
            biu_one_hot_encoding = torch.zeros(input_shape+(3,), dtype=torch.int, device=self.position_ids.device)

        if bbox is None:
            bbox = torch.zeros(tuple(list(input_shape) + [4]), dtype=torch.int, device=self.position_ids.device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        biu_embeddings = self.biu_embeddings(biu_one_hot_encoding.to(torch.float))

        embeddings = inputs_embeds + position_embeddings + token_type_embeddings + biu_embeddings

        if self.use_position_embeddings_2D:
            try:
                left_position_embeddings = self.x_position_embeddings(bbox[:, :, 0])
                upper_position_embeddings = self.y_position_embeddings(bbox[:, :, 1])
                right_position_embeddings = self.x_position_embeddings(bbox[:, :, 2])
                lower_position_embeddings = self.y_position_embeddings(bbox[:, :, 3])
                h_position_embeddings = self.h_position_embeddings(bbox[:, :, 3] - bbox[:, :, 1])
                w_position_embeddings = self.w_position_embeddings(bbox[:, :, 2] - bbox[:, :, 0])
            except IndexError as e:
                raise IndexError("The :obj:`bbox`coordinate values should be within 0-1000 range.") from e

            embeddings += left_position_embeddings \
                          + upper_position_embeddings \
                          + right_position_embeddings \
                          + lower_position_embeddings \
                          + h_position_embeddings \
                          + w_position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        
        return embeddings


# @TokenEmbedder.register("longformer-pretrained")
class PretrainedLongformerEmbedder(LongformerEmbedder):
    # pylint: disable=line-too-long
    """
    Parameters
    ----------
    pretrained_model: ``str``
        Either the name of the pretrained model to use (e.g. 'longformer-base-uncased'),
        or the path to the .tar.gz file with the model weights.

        If the name is a key in the list of pretrained models at
        https://github.com/huggingface/pytorch-pretrained-LONGFORMER/blob/master/pytorch_pretrained_longformer/modeling.py#L41
        the corresponding path will be used; otherwise it will be interpreted as a path or URL.
    requires_grad : ``bool``, optional (default = False)
        If True, compute gradient of LONGFORMER parameters for fine tuning.
    top_layer_only: ``bool``, optional (default = ``False``)
        If ``True``, then only return the top layer instead of apply the scalar mix.
    scalar_mix_parameters: ``List[float]``, optional, (default = None)
        If not ``None``, use these scalar mix parameters to weight the representations
        produced by different layers. These mixing weights are not updated during
        training.
    """
    def __init__(self, pretrained_model: str, requires_grad: str = "none", top_layer_only: bool = False,
                 scalar_mix_parameters: List[float] = None) -> None:
        
        model = PretrainedLongformerModel.load(pretrained_model)
        
        self._grad_layers = requires_grad

        if requires_grad in ["none", "all"]:
            for param in model.parameters():
                param.requires_grad = requires_grad == "all"
        else:
            model_name_regexes = requires_grad.split(",")
            for name, param in model.named_parameters():
                found = False
                for regex in model_name_regexes:
                    if regex in name:
                        found = True
                        break
                param.requires_grad = found

        super().__init__(longformer_model=model, top_layer_only=top_layer_only, scalar_mix_parameters=scalar_mix_parameters)


def create_position_ids_from_input_ids(input_ids, padding_idx):
    """
    Replace non-padding symbols with their position numbers. Position numbers begin at padding_idx+1. Padding symbols
    are ignored. This is modified from fairseq's `utils.make_positions`.

    Args:
        x: torch.Tensor x:

    Returns: torch.Tensor
    """
    # The series of casts and type-conversions here are carefully balanced to both work with ONNX export and XLA.
    mask = input_ids.ne(padding_idx).int()
    incremental_indices = torch.cumsum(mask, dim=1).type_as(mask) * mask
    return incremental_indices.long() + padding_idx