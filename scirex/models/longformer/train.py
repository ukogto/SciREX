

import logging
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
from sklearn.metrics import precision_score, recall_score, confusion_matrix
import math
import copy
import torch
import random
import sys
from dataclasses import dataclass, field
# from transformers import LongformerForMaskedLM, LongformerTokenizerFast, TextDataset, DataCollatorForLanguageModeling, Trainer
import numpy as np
from transformers import TrainingArguments, HfArgumentParser, LongformerConfig, EvalPrediction
from trainer import SMTrainer

from longformer.modeling_longformer import LongformerForPreTraining
from Datasets import SMDataset, SMValDataset, DataCollatorForSM
from tokenization import FullTokenizer

from transformers import LongformerTokenizer


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
import transformers
transformers.logging.set_verbosity_info()

def my_metrics(evalpred):
    
    predicted = evalpred.predictions
    labels = evalpred.label_ids
    # n, m = labels.shape
    if int(os.environ['EVAL_MODE']):
        acc = 100*np.sum(np.multiply((predicted == labels),(labels != -100)))/np.sum(labels != -100)
        print("accuracy : ", acc)
        return {'accuracy_mlm': acc }

    else:
        acc = 100*np.sum(np.multiply((predicted == labels),(labels != -100)))/np.sum(labels != -100)
        
        labels_f = copy.deepcopy(labels).flatten()
        predicted_f = copy.deepcopy(predicted).flatten()
        all_negative_index = [i for i in range(len(labels_f)) if labels_f[i] == -100]
        labels_f = np.delete(labels_f, all_negative_index)
        predicted_f = np.delete(predicted_f, all_negative_index)
        assert labels_f.shape == predicted_f.shape, "labels and predicted shape are not equal"
        ps = precision_score(labels_f, predicted_f)
        rc = recall_score(labels_f, predicted_f)
        # for i in range(len(predicted_f)):
        #     print(predicted_f[i], labels_f[i])
        
        print("precision score : ", ps, "recall score : ", rc , "accuracy : ", acc)
        tn, fp, fn, tp = confusion_matrix(labels_f, predicted_f).ravel()
        print("  tn, fp, fn, tp, num_of_labels, shape\n", tn, fp, fn, tp, np.sum(labels_f), labels.shape)
        
        return {'accuracy_ham': acc, 'precision score': ps, 'recall score' :  rc }


def create_model(save_model_to, attention_window, max_pos):
    #model = LongformerForPreTraining.from_pretrained('allenai/longformer-base-4096', gradient_checkpointing=True)
    model = LongformerForPreTraining.from_pretrained('allenai/longformer-base-4096') 
    config = model.config
    logger.info(f'saving model to {save_model_to}')
    model.save_pretrained(save_model_to)
    return model


def pretrain_and_evaluate(training_args, model_args, model, tokenizer_longformer, eval_only, model_path=None):
    training_args.mlm = model_args.mlm
    val_dataset = SMDataset(train_args=training_args, dir_path=model_args.val_datapath, instances_per_file=100, eval=True)
    if eval_only:
        train_dataset = val_dataset
    else:
        train_dataset = SMDataset(train_args=training_args, dir_path=model_args.train_datapath, instances_per_file=100)

    data_collator = DataCollatorForSM(tokenizer=tokenizer_longformer, mlm=model_args.mlm, args=model_args.__dict__)
    trainer = SMTrainer(model=model, args=training_args, data_collator=data_collator,
                      train_dataset=train_dataset, eval_dataset=val_dataset, compute_metrics=my_metrics)
    
    if not training_args.resume_from_checkpoint:
        print("***** Starting Evaluation *******")
        eval_loss = trainer.evaluate()
        eval_loss = eval_loss['eval_loss']
        print("Eval_loss : " ,eval_loss, "Eval_bpc : ", eval_loss/math.log(2))
        logger.info(f'Initial eval bpc: {eval_loss/math.log(2)}')

    if not eval_only:
        print("***** Starting Training *******")
        if model_path:
            trainer.train(model_path=model_path)
        else:
            trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        trainer.save_model()

        eval_loss = trainer.evaluate()
        eval_loss = eval_loss['eval_loss']
        logger.info(f'Eval bpc after pretraining: {eval_loss/math.log(2)}')
    else:
        trainer.evaluate()

@dataclass
class ModelArgs:
    attention_window: int = field(default=256, metadata={"help": "Size of attention window"})
    max_pos: int = field(default=1026, metadata={"help": "Maximum position"})
    config_file: str = field(default='longformer.config', metadata={"help": """The config json file corresponding to the pre-trained BERT model.
    This specifies the model architecture."""})
    vocab_file: str = field(default='vocab.txt', metadata={"help": ""})
    init_checkpoint: str = field(default=None, metadata={"help": "Initial checkpoint (usually from a pre-trained BERT model)."})
    masked_lm_prob: float = field(default=0.15, metadata={"help":"Masked LM probability."})
    short_seq_length: int = field(default=30, metadata={"help":"max no of tokens per node in bottom up creation"})
    deep_tree_prob: float = field(default=0.5, metadata={"help":"Probability with which to create deep tree instance from doc"})
    masked_node_prob: float = field(default=0.5, metadata={"help":"Probability with which an edge will be masked"})
    max_seq_length: int = field(default=1024, metadata={"help": "Max sequence length."})
    max_predictions_per_seq: int = field(default=20, metadata={"help": "Maximum number of masked LM predictions per sequence. Must match data generation."})
    max_num_headers: int = field(default=10, metadata={"help": "Maximum number of headers that will be present in the input"})
    max_tokens_per_header: int = field(default=30, metadata={"help": "Maximum number of tokens expected per header"})
    max_masked_nodes: int = field(default=5, metadata={"help": "Maximum number of nodes in the tree whose parent will be masked"})
    val_datapath: str = field(default='val/', metadata={"help": "Validation dataset path"})
    train_datapath: str = field(default='train/', metadata={"help": "Training dataset path"})
    mlm: int = field(default=-1, metadata={"help": """1 to pretrain on MLM task,
                                                      0 to pretrain on Header Alignment task,
                                                      -1 to pretrain on both task by choosing randomly for every training batch.
                                                      -10 to pretrain on both task by alternating each epoch"""})
    # eval_mode: int = field(default=0, metadata={"help": """1 to eval on MLM task,
                                                    #  0 to eval on Header Alignment task"""})


parser = HfArgumentParser((TrainingArguments, ModelArgs,))

training_args, model_args = parser.parse_args_into_dataclasses(look_for_args_file=True)
model_args.rng = random.Random(training_args.seed)
training_args.label_names = ["masked_lm_labels", "header_alignment_labels", "labels"]

# model_path = f'{training_args.output_dir}/longformer-base-{model_args.max_pos}'
# model_path = f'{training_args.output_dir}/longformer-base-2048/'
model_path = f'{training_args.output_dir}/longformer-base-1024-jan21/'

# if not os.path.exists(model_path):
#     os.makedirs(model_path)
#
# model, tokenizer = create_model(
#     save_model_to=model_path, attention_window=model_args.attention_window, max_pos=model_args.max_pos)
os.environ['EVAL_MODE'] = '2' # '0' = ham, '1' = mlm '2' both
os.environ["ALTERNATE"] = '1' #'0' = ham, '1' = mlm
logger.info(f'Loading the model from {model_path}')
# config = LongformerConfig()
#model = LongformerForPreTraining.from_pretrained('./output/longformer-base-2048-init/', gradient_checkpointing=False)
# model = LongformerForPreTraining.from_pretrained('allenai/longformer-base-4096') 
config = LongformerConfig.from_pretrained('../output/longformer-base-1024-mlm-ham-ps/')
model = LongformerForPreTraining.from_pretrained('../output/longformer-base-1024-mlm-ham-ps/', config = config)

# model = LongformerForPreTraining.from_pretrained('../output/longformer-base-1024-init/', config = config)

logger.info(f'Pretraining longformer-base-{model_args.max_pos} ... ')
# tokenizer_longformer = LongformerTokenizer.from_pretrained("./output/longformer-base-1024-init/")
#tokenizer_longformer = LongformerTokenizer.from_pretrained("allenai/longformer-base-4096")
tokenizer_longformer = LongformerTokenizer.from_pretrained("../allen_ai_longformer-base-4096")
if training_args.resume_from_checkpoint == "":
    print("From start")
    pretrain_and_evaluate(training_args, model_args, model, tokenizer_longformer, eval_only=False, model_path=model_path)
else:
    print("From checkpoint")
    pretrain_and_evaluate(training_args, model_args, model, tokenizer_longformer, eval_only=False)

logger.info(f'Saving model to {model_path}')
model.save_pretrained(model_path)

