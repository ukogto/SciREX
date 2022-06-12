from email import header
from transformers import Trainer, DataCollator, TrainingArguments, PreTrainedTokenizerBase, EvalPrediction, TrainerCallback
from transformers.modeling_utils import PreTrainedModel
from transformers.file_utils import is_sagemaker_mp_enabled
from transformers.trainer_pt_utils import nested_detach
import torch
import torch.nn.functional as F
from packaging import version
from torch import nn
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset, IterableDataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from torch.cuda.amp import autocast
import cbfl as focal
import collections
import inspect
import math
import os
import random
import re
import shutil
import sys
import tempfile
import time
import warnings
from logging import StreamHandler, log
from pathlib import Path, WindowsPath
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union


class SMTrainer(Trainer):
    def __init__(
        self,
        model: Union[PreTrainedModel, torch.nn.Module] = None,
        args: TrainingArguments = None,
        data_collator: Optional[DataCollator] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Dataset] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        model_init: Callable[[], PreTrainedModel] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
    ):
        super().__init__(model=model, args=args, data_collator=data_collator, train_dataset=train_dataset,
        eval_dataset=eval_dataset, tokenizer=tokenizer, model_init=model_init, compute_metrics=compute_metrics,
        callbacks=callbacks, optimizers=optimizers)

    

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step on :obj:`model` using obj:`inputs`.
        Subclass and override to inject custom behavior.
        Args:
            model (:obj:`nn.Module`):
                The model to evaluate.
            inputs (:obj:`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.
                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument :obj:`labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (:obj:`bool`):
                Whether or not to return the loss only.
            ignore_keys (:obj:`Lst[str]`, `optional`):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.
        Return:
            Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss,
            logits and labels (each being optional).
        """
        has_labels = any(inputs.get(k) is not None for k in self.label_names)
        inputs = self._prepare_inputs(inputs)
        if ignore_keys is None:
            if hasattr(self.model, "config"):
                ignore_keys = getattr(self.model.config, "keys_to_ignore_at_inference", [])
            else:
                ignore_keys = []

        # labels may be popped when computing the loss (label smoothing for instance) so we grab them first.
        if has_labels:
            labels = nested_detach(tuple(inputs.get(name) for name in self.label_names if inputs.get(name) is not None))
            if len(labels) == 1:
                labels = labels[0]
        else:
            labels = None

        with torch.no_grad():
            torch.cuda.empty_cache()
            if is_sagemaker_mp_enabled():
                raw_outputs = smp_forward_only(model, inputs)
                if has_labels:
                    if isinstance(raw_outputs, dict):
                        loss_mb = raw_outputs["loss"]
                        logits_mb = outputs.logits
                    else:
                        loss_mb = raw_outputs[0]
                        logits_mb = raw_outputs[1:]

                    loss = loss_mb.reduce_mean().detach().cpu()
                    logits = smp_nested_concat(logits_mb)
                else:
                    loss = None
                    if isinstance(raw_outputs, dict):
                        logits_mb = outputs.logits
                    else:
                        logits_mb = raw_outputs
                    logits = smp_nested_concat(logits_mb)
            else:
                if has_labels:
                    loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
                    loss = loss.mean().detach()
                    if isinstance(outputs, dict):
                        logits = outputs.logits
                    else:
                        logits = outputs[1:]
                else:
                    loss = None
                    if self.use_amp:
                        with autocast():
                            outputs = model(**inputs)
                    else:
                        outputs = model(**inputs)
                    if isinstance(outputs, dict):
                        logits = outputs.logits
                        loss = outputs.loss
                        loss = loss.mean().detach()
                    else:
                        logits = outputs
                        loss = outputs[0]
                        loss = loss.mean().detach()
                    # TODO: this needs to be fixed and made cleaner later.
                    if self.args.past_index >= 0:
                        self._past = outputs[self.args.past_index - 1]

        if prediction_loss_only:
            return (loss, None, None)
        
        logits = nested_detach(logits)
        _, logits = torch.max(logits, -1)
        
        # if len(logits) == 1:
        #     logits = logits[0]
        
        # print("********",logits,"*********", labels)
        
        return (loss, logits, labels)


def focal_cb_loss(ham_scores, header_alignment_labels, ignore_index):
    # _, logits = torch.max(ham_scores, -1)
    # indices = (header_alignment_labels == ignore_index).nonzero()
    # logits = logits[torch.arange(), indices]
    # labels = header_alignment_labels[indices]
    # print("1", header_alignment_labels.shape, ham_scores.shape)
    
    # print("number of ones ", torch.sum(header_alignment_labels == 1), torch.sum(header_alignment_labels == 0), torch.sum(header_alignment_labels == -100))
    indices = (header_alignment_labels != ignore_index).nonzero()
    logits = ham_scores[indices.T, :]
    # print("2", logits.shape, header_alignment_labels[indices.T].shape)
    logits = torch.squeeze(logits)
    # print(logits.shape)
    # logits = F.softmax(logits, dim=0)
    # print(logits)
    labels = torch.squeeze(header_alignment_labels[indices.T])
    # print("3", logits.shape, labels.shape)
    
    loss = focal.focal_loss(logits, labels)
    # loss = focal.fcbl(logits, labels)
    return(loss)
    

