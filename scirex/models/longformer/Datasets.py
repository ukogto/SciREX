import torch
from typing import Dict, List, Tuple, Optional, Any
import warnings
from dataclasses import dataclass
import json
import os
import natsort
import random
import time
import copy
from torch.utils.data.dataset import Dataset, IterableDataset
from create_pretraining_data import create_training_data_from_instance, create_instances_from_document
from inspect import signature


DEPRECATION_WARNING = (
    "This dataset will be removed from the library soon, preprocessing should be handled with the 🤗 Datasets "
    "library. You can have a look at this example script for pointers: {0}"
)

skip_data = 0
first_example = True
example_fail_safe = []
current_file = ""
class SMDataset(IterableDataset):
    """
    This will be superseded by a framework-agnostic approach soon.
    """

    def __init__(
        self,
        train_args,
        dir_path: str,
        # block_size: int,
        overwrite_cache=False,
        instances_per_file=100,
        eval=False,
        cache_dir: Optional[str] = None,
    ):
        super(SMDataset).__init__()
        warnings.warn(
            DEPRECATION_WARNING.format(
                "https://github.com/huggingface/transformers/blob/master/examples/pytorch/language-modeling/run_mlm.py"
            ),
            FutureWarning,
        )
        assert os.path.isdir(dir_path), f"Input dir path {dir_path} not found"
        self.dir_path = dir_path
        self.files = natsort.natsorted([name for name in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, name)) and name.endswith('.pkl')])
        self.eval = eval
        #calculating if skip_steps != -1 then skip dataloading till skip_step becomes 0 then make skip_step = -1
        global skip_data
        skip_data = 0
        if train_args.resume_from_checkpoint and not self.eval:
            prev_state_file = open(os.path.join(train_args.resume_from_checkpoint, "trainer_state.json"))
            state = json.load(prev_state_file)
            skip_data = 6120
        #skip_data = 0
        print("Total Files in Dataset:", len(self.files), "skip_steps = ",skip_data)
        self.instances_per_file = instances_per_file
        # block_size = block_size - tokenizer.num_special_tokens_to_add(pair=False)

        # directory, filename = os.path.split(file_path)
        # cached_features_file = os.path.join(
        #     cache_dir if cache_dir is not None else directory,
        #     f"cached_lm_{filename}",
        # )

        # Make sure only the first process in distributed training processes the dataset,
        # and the others will use the cache.
        # lock_path = cached_features_file + ".lock"
        # with FileLock(lock_path):
        #
        #     if os.path.exists(cached_features_file) and not overwrite_cache:
        #         start = time.time()
        #         with open(cached_features_file, "rb") as handle:
        #             self.examples = pickle.load(handle)
        #         logger.info(
        #             f"Loading features from cached file {cached_features_file} [took %.3f s]", time.time() - start
        #         )
        #
        #     else:
        #         logger.info(f"Creating features from dataset file at {directory}")
        #
        #         self.examples = []
        #         with open(file_path, encoding="utf-8") as f:
        #             text = f.read()
        #
        #         tokenized_text = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))
        #
        #         for i in range(0, len(tokenized_text) - block_size + 1, block_size):  # Truncate in block of block_size
        #             self.examples.append(
        #                 tokenizer.build_inputs_with_special_tokens(tokenized_text[i : i + block_size])
        #             )
        #         # Note that we are losing the last truncated example here for the sake of simplicity (no padding)
        #         # If your dataset is small, first you should look for a bigger one :-) and second you
        #         # can change this behavior by adding (model specific) padding.
        #
        #         start = time.time()
        #         with open(cached_features_file, "wb") as handle:
        #             pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)
        #         logger.info(
        #             f"Saving features into cached file {cached_features_file} [took {time.time() - start:.3f} s]"
        #         )

    def __len__(self):
        return len(self.files)*self.instances_per_file

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:  # multi-process data loading
            per_worker = int(math.ceil(len(self.files) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            iter_start = worker_id * per_worker
            iter_end = min(iter_start + per_worker, len(self.files))
            self.files = self.files[iter_start, iter_end]
        #global current_file
        global skip_data
        tfile = [{} for ii in range(self.instances_per_file)]#dummy data to be passed if skip is != 0
        for i, file in enumerate(self.files):
            #current_file = file
            if skip_data <= 0 or self.eval:
                tfile = torch.load(os.path.join(self.dir_path, file))
                #print(self.dir_path, file)
            #print(skip_data, tfile[0])
            # assert skip_data > 0 , [tfile[0]]
            skip_data = skip_data - 1
            #print(skip_data, i)
            for ind in range(self.instances_per_file):
                #ind_rm = ind
                if self.eval:
                    yield((tfile[ind], i*100+ind))
                else:
                    yield((tfile[ind], -1))


#    def __getitem__(self, i):
#        document = torch.load(os.path.join(self.dir_path, self.files[i//self.instances_per_file]))[i%self.instances_per_file]
#        return document, -1



class SMValDataset(Dataset):
    """
    This will be superseded by a framework-agnostic approach soon.
    """

    def __init__(
        self,
        dir_path: str,
        # block_size: int,
        overwrite_cache=False,
        instances_per_file=100,

        cache_dir: Optional[str] = None,
    ):
        warnings.warn(
            DEPRECATION_WARNING.format(
                "https://github.com/huggingface/transformers/blob/master/examples/pytorch/language-modeling/run_mlm.py"
            ),
            FutureWarning,
        )
        assert os.path.isdir(dir_path), f"Input dir path {dir_path} not found"
        self.dir_path = dir_path
        t0 = time.time()
        self.files = natsort.natsorted([name for name in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, name)) and name.endswith('.pkl')])
        self.files = [torch.load(os.path.join(self.dir_path, file)) for file in self.files]
        print(f"Time taken to load eval dataset: {(time.time()-t0):.2f}s")
        self.instances_per_file = instances_per_file
        # block_size = block_size - tokenizer.num_special_tokens_to_add(pair=False)

    def __len__(self):
        return len(self.files)*self.instances_per_file

    def get_file(self, i):
        return self.files[i//self.instances_per_file]

    def __getitem__(self, i):
        document = self.files[i//self.instances_per_file][i%self.instances_per_file]
#        document = torch.load(os.path.join(self.dir_path, self.files[i//self.instances_per_file]))[i%self.instances_per_file]
        return document, i






@dataclass
class DataCollatorForSM:
    """
    Data collator used for structured masking. Inputs are dynamically padded to the maximum length of a batch if they
    are not all of the same length.
    Args:
        tokenizer (:class:`~transformers.PreTrainedTokenizer` or :class:`~transformers.PreTrainedTokenizerFast`):
            The tokenizer used for encoding the data.
        mlm (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether or not to use masked language modeling. If set to :obj:`False`, the labels are the same as the
            inputs with the padding tokens ignored (by setting them to -100). Otherwise, the labels are -100 for
            non-masked tokens and the value to predict for the masked token.
        mlm_probability (:obj:`float`, `optional`, defaults to 0.15):
            The probability with which to (randomly) mask tokens in the input, when :obj:`mlm` is set to :obj:`True`.
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.
    .. note::
        For best performance, this data collator should be used with a dataset having items that are dictionaries or
        BatchEncoding, with the :obj:`"special_tokens_mask"` key, as returned by a
        :class:`~transformers.PreTrainedTokenizer` or a :class:`~transformers.PreTrainedTokenizerFast` with the
        argument :obj:`return_special_tokens_mask=True`.
    """

    # tokenizer: PreTrainedTokenizerBase
    tokenizer: Any = None
    mlm: int = -1 # -1 for random, 0 for header alignment, 1 for mlm
    pad_to_multiple_of: Optional[int] = None
    args: Dict = None

    def __post_init__(self):
        if self.mlm not in [-1, 0, 1, -10]:
            raise Exception(f"mlm should be in -1, 0 or 1 but given is {self.mlm}")
        self.rng = self.args.pop("rng")
        f1sig = signature(create_instances_from_document).parameters.keys()
        f2sig = signature(create_training_data_from_instance).parameters.keys()
        self.f1inps = {k:self.args[k] for k in f1sig if k not in ['document', 'tokenizer', 'rng']}
        self.f2inps = {k:self.args[k] for k in f2sig if k not in ['doc', 'tokenizer', 'rng']}

        self.f2inps.update({'tokenizer': self.tokenizer})
        self.pad_token_id = self.tokenizer.convert_tokens_to_ids(["<pad>"])[0]

    def __call__(
        self, examples
    ) -> Dict[str, torch.Tensor]:
        train = True
        if self.mlm == -1:
            is_mlm = torch.rand(1)[0] < 0.5
        elif self.mlm == -10:
            is_mlm = int(os.environ['ALTERNATE'])
        else:
            is_mlm = self.mlm
        if examples[0][1] != -1:
            train = False
            # if self.mlm not in [0,1]:
            is_mlm = int(os.environ['EVAL_MODE'])
            examples = [create_training_data_from_instance(create_instances_from_document(e, rng=random.Random(i), **self.f1inps), rng=random.Random(i), **self.f2inps).__dict__ for e, i in examples]
        else:
            examples = [create_training_data_from_instance(create_instances_from_document(e, rng=self.rng, **self.f1inps), rng=self.rng, **self.f2inps).__dict__ for e, i in examples]
        
       
        # for each_item in range(len(examples)):
        #     print(current_file, examples[each_item]["token_type_ids"][-1], examples[each_item]["input_ids"].shape)
        # ###################################
        # global first_example
        # global example_fail_safe
        # if first_example:
        #    example_fail_safe = copy.deepcopy(examples)
        #    first_example = False
        # # print(example_fail_safe)
        # for each_item in range(len(examples)):
        #    if examples[each_item]["token_type_ids"][-1] > 255 or examples[each_item]["input_ids"].shape[0] > 1025:
        #     #    print(examples[each_item]["token_type_ids"][-1], examples[each_item]["input_ids"].shape)
        #        temp = copy.deepcopy(example_fail_safe[0])
        #        examples[each_item] = temp
        # ################################
        
        for e in examples:
            if is_mlm:
                e.pop('header_positions')
                e.pop('masked_header_positions')
                e.pop('header_alignment_labels')
                e.pop('header_alignment_weights')
            else:
                e.pop('masked_lm_labels')
        
        # Check if padding is necessary.
        ham_keys = set(['header_positions', 'masked_header_positions', 'header_alignment_labels', 'header_alignment_weights' ])
        e0keys = set(examples[0].keys()) - ham_keys
        length_of_first = {k: examples[0][k].size(0) for k in e0keys}
        are_tensors_same_length = {k:all(x[k].size(0) == length_of_first[k] for x in examples) for k in e0keys}

        # If yes, check if we have a `pad_token`.
        if self.pad_token_id is None:
            raise ValueError(
                "You are attempting to pad samples but pad_token_id is None"
            )

        # Creating the full tensor and filling it with our data.
        mg = self.args["max_num_headers"]*self.args["max_tokens_per_header"]
        batch = {'is_mlm':torch.full((len(examples),1), 1, dtype=torch.int)}
        if not is_mlm:
            for k in ham_keys:
                batch[k] = torch.stack([e[k] for e in examples])
        for k in e0keys:
            max_length = max(x[k].size(0) for x in examples)
            if self.pad_to_multiple_of is not None and (max_length % self.pad_to_multiple_of != 0):
                max_length = ((max_length // self.pad_to_multiple_of) + 1) * self.pad_to_multiple_of
            fill_val = self.pad_token_id if k == "input_ids" else -100 if "label" in k else 0
            shape = examples[0][k].shape
            if k == "mask_global":
                shape = (len(examples), max_length, mg)
            else:
                shape = (len(examples), max_length, ) + shape[1:]
            batch[k] = examples[0][k].new_full(shape, fill_val)
            for i, example in enumerate(examples):
                if k == "mask_global":
                    batch[k][i, : example[k].shape[0], :example[k].shape[1]] = example[k]
                else:
                    batch[k][i, : example[k].shape[0]] = example[k]

        
        # index = torch.ones(batch["mask_local"].shape[2], dtype=bool)
        # index[torch.max(torch.sum(batch["global_attention_mask"], dim=1)):mg] = False
        # print(batch["global_attention_mask"].sum(dim=1).max())
        # batch["mask_local"] = batch["mask_local"][:, :, index]
        print((batch["input_ids"][0]==2).nonzero(as_tuple=True)[0])
        for k in batch:
           print(k, batch[k].shape)
        print("***************")
        return batch

