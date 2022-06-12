# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Create masked LM/next sentence masked_lm TF examples for BERT."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import random
from scirex.models.longformer import tokenization
import numpy as np
import json
import torch
import argparse
import os

# class PklTrainingInstance(object):
#   def __init__(self, input_ids, formats, bboxes, input_mask, segment_ids, masked_lm_positions, masked_lm_ids, masked_lm_weights,
#                masked_header_weights,masked_header_labels,local_attention_matx, global_attention_mask,global_attention_matx,
#                header_positions,masked_node_positions):
#     self.input_ids = torch.tensor(input_ids, dtype=torch.long)
#     self.formats=torch.Tensor(formats)
#     self.bboxes=torch.Tensor(bboxes)
#     self.input_mask=torch.Tensor(input_mask)
#     self.segment_ids = torch.Tensor(segment_ids)
#     self.masked_lm_positions = torch.Tensor(masked_lm_positions)
#     self.masked_lm_labels = torch.Tensor(masked_lm_ids)
#     self.masked_lm_weights=torch.Tensor(masked_lm_weights)
#     self.masked_header_weights=torch.Tensor(masked_header_weights)
#     self.masked_header_labels=torch.Tensor(masked_header_labels)
#     self.global_attention_mask=torch.Tensor(global_attention_mask)
#     self.local_attention_matrix=torch.Tensor(local_attention_matx)
#     self.global_attention_matrix=torch.Tensor(global_attention_matx)
#     self.header_positions=torch.Tensor(header_positions)
#     self.masked_node_positions=torch.Tensor(masked_node_positions)

class TrainingInstance(object):
  """A single training instance (sentence pair)."""

  def __init__(self, tokens, formats, bboxes, segment_ids, attention_mask, masked_lm_labels,
               # masked_header_weights,masked_header_labels,global_attention_mask,attention_matx,
               masked_header_weights,masked_header_labels,local_attention_matx,
               global_attention_mask,global_attention_matx,
               header_positions,masked_node_positions):
    self.input_ids = torch.Tensor(tokens).to(torch.long)
    for i in range(len(formats)):
        if len(formats[i]) == 2:
            formats[i].append(0)
    self.biu_one_hot_encoding=torch.Tensor(formats).to(torch.int)
    # self.biu_one_hot_encoding=torch.zeros((len(tokens), 3), dtype=torch.long)
    self.bbox=torch.Tensor(bboxes).to(torch.int)
    self.token_type_ids = torch.Tensor(segment_ids).to(torch.long)
    self.attention_mask = torch.Tensor(attention_mask)
    self.masked_lm_labels = torch.Tensor(masked_lm_labels).to(torch.long)
    self.header_alignment_weights=torch.Tensor(masked_header_weights)
    self.header_alignment_labels=torch.Tensor(masked_header_labels).to(torch.long)
    self.mask_local=torch.Tensor(local_attention_matx).to(torch.int)
    self.global_attention_mask=torch.Tensor(global_attention_mask).to(torch.int)
    self.mask_global=torch.Tensor(global_attention_matx).to(torch.int)
    # self.attention_matx=attention_matx
    self.header_positions=torch.Tensor(header_positions).to(torch.int)
    self.masked_header_positions=torch.Tensor(masked_node_positions).to(torch.int)

  def __str__(self):
    s = ""
    s += "tokens: %s\n" % (" ".join(
        [str(x) for x in self.tokens]))
    s += "formats: %s\n" % (" ".join(
        [str(x) for x in self.formats]))
    s += "bboxes: %s\n" % (" ".join(
        [str(x) for x in self.bboxes]))
    s += "segment_ids: %s\n" % (" ".join([str(x) for x in self.segment_ids]))
    s += "masked_lm_positions: %s\n" % (" ".join(
        [str(x) for x in self.masked_lm_positions]))
    s += "masked_lm_labels: %s\n" % (" ".join(
        [tokenization.printable_text(x) for x in self.masked_lm_labels]))
    s += "attention_matx: %s\n" % ("\n".join(
      [" ".join([str(x) for x in y]) for y in self.attention_matx]))
    s += "header_positions: %s\n" % (" ".join(
        [str(x) for x in self.header_positions]))
    s += "masked_node_positions: %s\n" % (" ".join(
        [str(x) for x in self.masked_node_positions]))
    s += "masked_header_weights: %s\n" % ("\n".join(
      [" ".join([str(x) for x in y]) for y in self.masked_header_weights]))
    s += "masked_header_labels: %s\n" % ("\n".join(
      [" ".join([str(x) for x in y]) for y in self.masked_header_labels]))
    s += "\n"
    return s

  def __repr__(self):
    return self.__str__()

class TrainingInstanceForEndTask(object):
  """A single training instance (sentence pair)."""

  def __init__(self, tokens, formats, bboxes, segment_ids, attention_mask, local_attention_matx, global_attention_mask, global_attention_matx):
    self.input_ids = torch.Tensor(tokens).to(torch.long)
    for i in range(len(formats)):
        if len(formats[i]) == 2:
            formats[i].append(0)
    self.biu_one_hot_encoding=torch.Tensor(formats).to(torch.int)
    # self.biu_one_hot_encoding=torch.zeros((len(tokens), 3), dtype=torch.long)
    self.bbox=torch.Tensor(bboxes).to(torch.int)
    self.token_type_ids = torch.Tensor(segment_ids).to(torch.long)
    self.attention_mask = torch.Tensor(attention_mask)
    self.mask_local=torch.Tensor(local_attention_matx).to(torch.int)
    self.global_attention_mask=torch.Tensor(global_attention_mask).to(torch.int)
    self.mask_global=torch.Tensor(global_attention_matx).to(torch.int)
    # self.attention_matx=attention_matx

  def __str__(self):
    s = ""
    s += "tokens: %s\n" % (" ".join(
        [str(x) for x in self.tokens]))
    s += "formats: %s\n" % (" ".join(
        [str(x) for x in self.formats]))
    s += "bboxes: %s\n" % (" ".join(
        [str(x) for x in self.bboxes]))
    s += "segment_ids: %s\n" % (" ".join([str(x) for x in self.segment_ids]))
    s += "attention_matx: %s\n" % ("\n".join(
      [" ".join([str(x) for x in y]) for y in self.attention_matx]))
    s += "\n"
    return s

  def __repr__(self):
    return self.__str__()

# def write_instance_to_example_files(instances, tokenizer, max_seq_length,
#                                     max_predictions_per_seq, output_dir, written_offset):
#   """Create TF example files from `TrainingInstance`s."""
#   # writers = []
#   # for output_file in output_files:
#   #   writers.append(tf.python_io.TFRecordWriter(output_file))
#   if not os.path.isdir(output_dir):
#     os.mkdir(output_dir)

#   final_instances=[]

#   writer_index = 0

#   total_written = 0
#   for (inst_index, instance) in enumerate(instances):
#     input_ids = tokenizer.convert_tokens_to_ids(instance.tokens)
#     input_mask = [1] * len(input_ids)
#     segment_ids = list(instance.segment_ids)
#     this_format=[]
#     for form in instance.formats:
#       this_format.extend(form)
#       # this_format.append(0) ##### no underline
#     this_bbox=[]
#     for bbox in instance.bboxes:
#       this_bbox.extend(bbox)
#     assert len(input_ids) <= max_seq_length

#     while len(input_ids) < max_seq_length:
#       input_ids.append(0)
#       input_mask.append(0)
#       segment_ids.append(0)
#       this_format.extend([0,0,0])
#       this_bbox.extend([0,0,0,0])

#     assert len(input_ids) == max_seq_length
#     assert len(input_mask) == max_seq_length
#     assert len(segment_ids) == max_seq_length

#     masked_lm_positions = list(instance.masked_lm_positions)
#     masked_lm_ids = tokenizer.convert_tokens_to_ids(instance.masked_lm_labels)
#     masked_lm_labels = -100*np.ones((max_seq_length, ))
#     masked_lm_labels[masked_lm_positions] = masked_lm_ids
#     masked_lm_weights = np.zeros((max_seq_length, ))
#     masked_lm_weights[masked_lm_positions] = 1.0

#     while len(masked_lm_positions) < max_predictions_per_seq:
#       masked_lm_positions.append(0)
#       # masked_lm_ids.append(0)
#       # masked_lm_weights.append(0.0)

#     # print(type(masked_lm_weights))
#     # print(type(instance.masked_header_weights))
#     # print(len(instance.masked_header_weights))

#     final_instances.append(PklTrainingInstance(input_ids,this_format,this_bbox,input_mask,segment_ids,masked_lm_positions,masked_lm_labels,
#       masked_lm_weights,instance.masked_header_weights,instance.masked_header_labels,instance.local_attention_matx,instance.global_attention_mask,
#       instance.global_attention_matx,instance.header_positions,instance.masked_node_positions))

#     # features = collections.OrderedDict()
#     # features["input_ids"] = create_int_feature(input_ids)
#     # features["formats"]=create_int_feature(this_format)
#     # features["bboxes"]=create_int_feature(this_bbox)
#     # features["input_mask"] = create_int_feature(input_mask)
#     # features["segment_ids"] = create_int_feature(segment_ids)
#     # features["masked_lm_positions"] = create_int_feature(masked_lm_positions)
#     # features["masked_lm_ids"] = create_int_feature(masked_lm_ids)
#     # features["masked_lm_weights"] = create_float_feature(masked_lm_weights)
#     # features["masked_header_weights"]= create_float_feature(instance.masked_header_weights)
#     # features["masked_header_labels"]= create_int_feature(instance.masked_header_labels)
#     # features["local_attention_matx"]= create_float_feature(instance.local_attention_matx)
#     # features["global_attention_mask"]= create_float_feature(instance.global_attention_mask)
#     # features["global_attention_matx"]= create_float_feature(instance.global_attention_matx)
#     # # features["attention_matx"]= create_float_feature(instance.attention_matx)
#     # features["header_positions"]= create_int_feature(instance.header_positions)
#     # features["masked_node_positions"]= create_int_feature(instance.masked_node_positions)
#     #
#     #
#     #
#     # tf_example = tf.train.Example(features=tf.train.Features(feature=features))
#     #
#     # writers[writer_index].write(tf_example.SerializeToString())
#     # writer_index = (writer_index + 1) % len(writers)
#     if len(final_instances) == 1000:
#         torch.save(final_instances, output_dir+str(int(written_offset)+total_written)+'.pkl')
#         total_written += 1000
#         final_instances = []

#   #   if inst_index < 20:
#   #     print("*** Example ***")
#   #     print("tokens: %s" % " ".join(
#   #         [tokenization.printable_text(x) for x in instance.tokens]))
#   #
#   #     for feature_name in features.keys():
#   #       if feature_name in ["attention_matx"]:
#   #         continue
#   #       feature = features[feature_name]
#   #       values = []
#   #       if feature.int64_list.value:
#   #         values = feature.int64_list.value
#   #       elif feature.float_list.value:
#   #         values = feature.float_list.value
#   #       print.info(
#   #           "%s: %s" % (feature_name, " ".join([str(x) for x in values])))
#   #
#   # for writer in writers:
#   #   writer.close()

#   print("Total instances written=", total_written)
#   print("Total instances over all runs=", int(written_offset)+total_written)


def create_int_feature(values):
  feature = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
  return feature


def create_float_feature(values):
  feature = tf.train.Feature(float_list=tf.train.FloatList(value=list(values)))
  return feature


def get_level(line):
  """returns a list of indices to get to the level in doc where line is to be added"""
  levs,line=line.split('#',1)
  levs=levs.split()
  ind_list=[]
  
  for l in levs:
    try:
      if int(l)==0:
        ind_list.append(0)
      else:
        ind_list.append(int(l)-1)
    except:
      ind_list.append('P')

  return (int(levs[0]), ind_list,line)

def save_docs(docs,output_dir,written_offset,total_written):
  if not os.path.isdir(output_dir):
    os.mkdir(output_dir)

  torch.save(docs, output_dir+str(int(written_offset)+total_written)+'.pkl')
  # total_written += len(instances)

  # return total_written


def create_training_instances(input_files, tokenizer, max_seq_length,
                              dupe_factor, short_seq_length, masked_lm_prob, deep_tree_prob,masked_node_prob,
                              max_num_headers, max_tokens_per_header,max_masked_nodes,max_predictions_per_seq, rng,
                              output_dir,written_offset):
                              # max_num_headers, max_masked_nodes,max_predictions_per_seq, rng):
  """Create `TrainingInstance`s from raw text."""

  # each level is a dict of type {"title":,"content":[],"sublevels":[]}, document is a dict of the topmost level
  all_documents = []

  # Input file format: txt and output is in json which store as a pickle file
  # (1)
  #    1#Doc Title
  #    1 P#Content
  #    1 1#Header
  #    1 1 P#Par
  #    1 1 1#Sub-header
  #    1 1 1 P#Par
  #    1 1 1 P#Par
  #    1 1 2#Sub-header
  #    2#Document2 title
  # (2) Keep multiple documents within same file to reduce seek time
  # (3) Word sequence is written as: word1 fontinfo1 bbox1 word2 fontinfo2 bbox2 ...
  for input_file in input_files:
    print("reading input file: ",input_file, "number of input files: ", len(input_files))
    with open(input_file, "r") as reader:
    
      lines_done=0
      
      while True:

        line = tokenization.convert_to_unicode(reader.readline())
        if not line:
          break
        line = line.strip()

        # # Empty lines are used as document delimiters
        # if not line:
        #   all_documents.append([])

        lines_done+=1
        if lines_done%10000==0:
          print(lines_done," lines done")
        # determine level and the actual line
        (article_num, tree_ind_list, line)=get_level(line)
        tokens = tokenizer.tokenize(line) # list of list of token, list of format, list of bbox
        if tokens:
          # is_content=False
          #tokens[0]=tokenizer.convert_tokens_to_ids(tokens[0]) #already converted in new code
          tokens[1]=[[int(x=="True") for x in tokens[1][token_ind].split(",")] for token_ind in range(len(tokens[1]))]
          tokens[2]=[[int(x) for x in tokens[2][token_ind].split(",")] for token_ind in range(len(tokens[2]))]
          level_list=all_documents
          
          for (ind,i) in enumerate(tree_ind_list):
            if i=='P':
              # content of curr_dict
              if "content" not in curr_dict:
                curr_dict["content"]=[]

              curr_dict["content"].append(tokens)
              break
            
            while i>=len(level_list):
              # push empty dicts till i<len
              level_list.append({})

            curr_dict=level_list[i]
            if ind+1<len(tree_ind_list):#either content or go deeper
              if tree_ind_list[ind+1]!='P': # go deeper
                if "sublevels" not in curr_dict:
                  curr_dict["sublevels"]=[]
                level_list=curr_dict["sublevels"]
            else: # not P and no other indices means it is title of this dict
              curr_dict["title"]=tokens
        
        # if int(tree_ind_list[0]) >= 3000:
        #   print(tree_ind_list[0], len(all_documents))
        #   break
  # print(json.dumps(all_documents,indent="\t"))
  

          # all_documents[-1].append(tokens)
  # Remove empty documents
  all_documents = [x for x in all_documents if x]
  rng.shuffle(all_documents)
  

  print("no docs=",len(all_documents))

  total_written=0

  write_docs=[]

  for doc in all_documents:
    if doc=={}:# empty
      print("found empty document!!")
      exit()
    write_docs.append(doc)
    if len(write_docs)==100:
      save_docs(write_docs,output_dir,written_offset,total_written)
      write_docs=[]
      total_written+=100

  print("Total instances written=", total_written)
  print("Total instances over all runs=", int(written_offset)+total_written)



  # vocab_words = list(tokenizer.vocab.keys())
  # instances = []
  # for _ in range(dupe_factor):
  #   for document_index in range(len(all_documents)):
  #     if all_documents[document_index]=={}:# empty
  #       print("found empty document!!")
  #       exit()
  #     found_skip=False
  #     for skip_word in ['image','talk','user','category','template']:
  #       if skip_word in all_documents[document_index]["title"][0]:
  #         found_skip=True
  #         break
  #     if found_skip:
  #       print("skipping title ",all_documents[document_index]["title"][0])
  #       continue
  #     instances.append(
  #         create_instances_from_document(
  #             all_documents, document_index, max_seq_length, short_seq_length,
  #             deep_tree_prob,masked_lm_prob, masked_node_prob,max_num_headers, max_tokens_per_header, max_masked_nodes,
  #             # deep_tree_prob,masked_lm_prob, masked_node_prob,max_num_headers, max_masked_nodes,
  #             max_predictions_per_seq, vocab_words, rng))

  #     if len(instances)==1000:
  #       save_instances(instances,output_dir,written_offset,total_written)
  #       total_written+=1000
  #       instances=[]

  # print("Total instances written=", total_written)
  # print("Total instances over all runs=", int(written_offset)+total_written)
  # rng.shuffle(instances)
  # return instances
####################### text to pickle processing function ends here ############################################


def create_doc_deep_tree(ip_node,curr_node, rng,max_seq_length,short_seq_length,
  max_num_headers,max_tokens_per_header,no_headers_used,no_pars,toks_used):
  # max_num_headers,no_headers_used,no_pars,toks_used):
  """create as deep a tree from the doc as possible
  at each level dont use a paragraph bigger than short_seq_length in length"""

  # toks_used=no title tokens in all headers above this level

  if no_headers_used>=max_num_headers or toks_used>=max_seq_length:
    # print("no_headers_used=",no_headers_used)
    # print("toks_used=",toks_used)
    return ip_node,no_headers_used,no_pars,toks_used


  ip_title=curr_node.get("title",[[],[],[]]).copy() # even if this node has empty title, still add
  # restrict to max_tokens_per_header tokens
  if len(ip_title[0])>max_tokens_per_header:
    for i in range(3):
      ip_title[i]=ip_title[i][:max_tokens_per_header]
  ip_node["title"]=ip_title
  toks_used+=len(ip_node["title"][0])
  no_headers_used+=1

  child_nodes=curr_node.get("sublevels",[]).copy()

  rng.shuffle(child_nodes)

  ip_node["sublevels"]=[]

  for node in child_nodes:
    ip_node["sublevels"].append({})
    ip_node["sublevels"][-1],new_no_headers_used,no_pars,toks_used=create_doc_deep_tree(ip_node["sublevels"][-1],node,rng,
      max_seq_length,short_seq_length,max_num_headers,max_tokens_per_header, no_headers_used,no_pars,toks_used)
      # max_seq_length,short_seq_length,max_num_headers, no_headers_used,no_pars,toks_used)
    if new_no_headers_used==no_headers_used:
      del ip_node["sublevels"][-1]
    no_headers_used=new_no_headers_used

  # tree for subtree rooted at curr_node has been created, now add it's contents

  ip_node["content"]=[]

  for cont in curr_node.get("content",[]):
    if toks_used+no_headers_used+no_pars>=max_seq_length:
      break
    else:
      chosen_sublist=rng.randint(0,min(short_seq_length,len(cont[0])))
      toks_used+=chosen_sublist
      no_pars+=1
      ip_node["content"].append([cont[0][:chosen_sublist],cont[1][:chosen_sublist],cont[2][:chosen_sublist]])

  # print(json.dumps(curr_node,indent="\t"))
  # print(json.dumps(ip_node,indent="\t"))
  return ip_node,no_headers_used,no_pars,toks_used




# def create_doc_broad_tree(ip_level,curr_level,rng,max_seq_length,max_num_headers,no_headers,no_pars,toks_used):
def create_doc_broad_tree(ip_level,curr_level,rng,max_seq_length,max_num_headers,max_tokens_per_header,no_headers,no_pars,toks_used):
  """create a tree as broad as possible from the doc
  ip_level: list of dicts to be made for nodes at the present level
  curr_level: list of dicts of nodes from which ip_level is to be made
  """

  if curr_level==[] or toks_used+no_headers+no_pars>=max_seq_length:
    return ip_level,no_headers,no_pars,toks_used

  # rng.shuffle(curr_level)


  for i in range(len(curr_level)):
    # add curr_level[i]'s info to ip_level[i]
    if toks_used+no_headers+no_pars>=max_seq_length or no_headers>=max_num_headers:
      break

    ip_title=curr_level[i].get("title",[[],[],[]]).copy()
    # restrict to max_tokens_per_header tokens
    if len(ip_title[0])>max_tokens_per_header:
      for j in range(3):
        ip_title[j]=ip_title[j][:max_tokens_per_header]
    ip_level[i]["title"]=ip_title
    no_headers+=1
    toks_used+=len(ip_level[i]["title"][0])

    ip_level[i]["content"]=[]
    for cont in curr_level[i].get("content",[]):
      if toks_used+no_headers+no_pars>=max_seq_length:
        break
      ip_level[i]["content"].append(cont)
      no_pars+=1
      toks_used+=len(ip_level[i]["content"][-1][0])

  # all nodes in curr_level were added

  node_ind_list=list(range(len(curr_level)))

  if no_headers>=max_num_headers:
    # no need to add sublevels
    return ip_level,no_headers,no_pars,toks_used

  rng.shuffle(node_ind_list) # shuffle nodes at curr_level to generate next_level

  next_level=[]
  gen_next_level=[]

  subnode_ind_list=[[] for x in range(len(curr_level))]

  for index in node_ind_list:
    if len(curr_level[index].get("sublevels",[]))>0:
      ip_level[index]["sublevels"]=[]
    subnode_ind_list[index]=list(range(len(curr_level[index].get("sublevels",[]))))
    rng.shuffle(subnode_ind_list[index]) # shuffle children of curr_level[index]

    for subid in subnode_ind_list[index]:
      ip_level[index]["sublevels"].append({})
      next_level.append(curr_level[index]["sublevels"][subid])
      # gen_next_level.append(ip_level[index]["sublevels"][-1])
      gen_next_level.append({})

  gen_next_level,no_headers,no_pars,toks_used=create_doc_broad_tree(gen_next_level,next_level,rng,
    max_seq_length,max_num_headers,max_tokens_per_header, no_headers,no_pars,toks_used)
    # max_seq_length,max_num_headers,no_headers,no_pars,toks_used)

  gen_next_ind=0
  for index in node_ind_list:
    for subid in subnode_ind_list[index]:
      ip_level[index]["sublevels"][subid]=gen_next_level[gen_next_ind]
      gen_next_ind+=1
    if "sublevels" in ip_level[index]:
      ip_level[index]["sublevels"]=list(filter(lambda x: x!={},ip_level[index]["sublevels"]))

  # print(json.dumps(curr_level,indent="\t"))
  # print(json.dumps(ip_level,indent="\t"))

  return ip_level,no_headers,no_pars,toks_used


def create_instances_from_document(
    document, max_seq_length, short_seq_length, deep_tree_prob,
    masked_lm_prob, masked_node_prob, max_num_headers, max_tokens_per_header, max_masked_nodes, max_predictions_per_seq, rng):
    # masked_lm_prob, masked_node_prob, max_num_headers, max_masked_nodes, max_predictions_per_seq, vocab_words, rng):
  """Creates `TrainingInstance`s for a single document."""

  # document = all_documents[document_index]

  # either create a deep or a broad tree
  if rng.random()<deep_tree_prob:
    # print("creating a deep tree")
    doc,no_headers,no_pars,toks_used=create_doc_deep_tree(
      {},document,rng,max_seq_length,short_seq_length,max_num_headers,max_tokens_per_header,0,0,0)
      # {},document,rng,max_seq_length,short_seq_length,max_num_headers,0,0,0)
  else:
    # print("creating broad tree")
    doc,no_headers,no_pars,toks_used=create_doc_broad_tree(
      [{}],[document],rng,max_seq_length,max_num_headers,max_tokens_per_header,0,0,0)
      # [{}],[document],rng,max_seq_length,max_num_headers,0,0,0)
    doc=doc[0]

  # print(json.dumps(doc,indent="\t"))

  assert(no_headers<=max_num_headers)

  return doc

def create_training_data_from_instance(doc,attention_window,max_seq_length,masked_node_prob,max_masked_nodes,masked_lm_prob,
  max_num_headers,max_tokens_per_header,max_predictions_per_seq,tokenizer,rng):

  token_list=[]
  format_list=[]
  bbox_list=[]
  no_toks=0
  token_list,format_list,bbox_list,_,attention_dict,no_toks,no_masks, samples,masked_edges,header_pos,ancestor_dict=create_nodes_list(
    doc,{},max_seq_length,token_list,format_list,bbox_list,no_toks,[],[],[],{},
    {},masked_node_prob,max_masked_nodes,0,[],tokenizer,rng)

  if no_masks == 0:
    #print("samples", samples)
    num_mask = min(rng.randint(1, 3), len(samples))
    mask_edges_sample = rng.sample(samples, num_mask)
    
    for each_edge in mask_edges_sample:
      masked_edges[each_edge[0]]=each_edge[1]
      attention_dict[each_edge[1]].remove(each_edge[0])
      if each_edge[-1] == "L":#if leaf
        attention_dict[each_edge[0]]=[]
      no_masks += 1
  
  #print(masked_edges, len(samples), no_masks)
  # assert no_masks != 0 and len(mask_edges_sample) != 0, [num_mask, mask_edges_sample]
  
  masked_ids=[]
  masked_parent_ids=[]
  for masked_id in masked_edges:
    masked_ids.append(masked_id)
    masked_parent_ids.append(masked_edges[masked_id])
  # print(ancestor_dict, masked_edges, masked_parent_ids, masked_ids)
  # print(token_list)
  # print(format_list)
  # print(bbox_list)
  # print(attention_dict)
  # print(no_toks,no_masks)
  # print(masked_edges)
  # print(masked_ids)
  # print(masked_parent_ids)
  # print(header_pos)
  # print(ancestor_dict)

  tokens_index_list=list(range(len(token_list))) # shuffle obtained nodes

  rng.shuffle(tokens_index_list)
  len_tokens = sum(map(len, token_list))
  # local_attention_matx=np.zeros((len_tokens,len_tokens)) # attention local to the node

  # global_attention_matx=np.zeros((len_tokens,max_num_headers*max_tokens_per_header)) # attention for header nodes

  global_attention_mask=np.zeros((len_tokens,), dtype=int)
  attention_matx=np.zeros((len_tokens,len_tokens))


  tokens=[]
  formats=[]
  bboxes=[]
  segment_ids=[]
  header_positions=[]
  masked_node_positions=[]

  masked_header_labels=np.zeros((max_num_headers,max_masked_nodes),dtype=int)
  masked_header_weights=np.zeros((max_num_headers,max_masked_nodes))

  index_positions=np.zeros((len(token_list),2),dtype=int)

  header_numbering={}
  no_header_tokens=0

  for s_id,node_ind in enumerate(tokens_index_list):
    start_pos=len(tokens)
    index_positions[node_ind][0]=start_pos
    for tok_ind in range(len(token_list[node_ind])):
      tokens.append(token_list[node_ind][tok_ind])
      formats.append(format_list[node_ind][tok_ind])
      bboxes.append(bbox_list[node_ind][tok_ind])
      segment_ids.append(s_id)
    end_pos=len(tokens)-1
    index_positions[node_ind][1]=end_pos

    # if end_pos+1==start_pos:
    #   index_positions[node_ind][0]=-1
    #   index_positions[node_ind][1]=-1
    #   # print("had to remove node: ",node_ind)
    #   assert(False)
    #   continue

    assert(end_pos+1>start_pos)

    if node_ind in header_pos:
      # print("header node")
      # print(tokens[start_pos:end_pos+1])
      # print("start_pos=",start_pos)
      # print("end_pos=",end_pos)
      header_positions.append(start_pos)
      global_attention_mask[start_pos:end_pos+1]=1
      # print("global_attention_mask sublist:",global_attention_mask[start_pos:end_pos+1])
      header_numbering[node_ind]=[no_header_tokens,no_header_tokens+end_pos-start_pos]
      # print("header_numbering: ",header_numbering[node_ind])
      no_header_tokens+=1+end_pos-start_pos

    # self-attention
    # local_attention_matx[start_pos:end_pos+1,start_pos:end_pos+1]=1
    attention_matx[start_pos:end_pos+1,start_pos:end_pos+1]=1

  # print(tokens_index_list)
  # print(tokens)
  # print(index_positions)
  # print(list(attention_matx))

  # attention by tree structure
  # restrict to ancestors
  for node_ind in tokens_index_list:
    for att_ind in attention_dict.get(node_ind,[]):
      # attend to att_ind
      # print("node_ind=",node_ind)
      # print("att_ind=",att_ind)
      # print(index_positions[node_ind][0])
      # print(index_positions[node_ind][1]+1)
      # print(index_positions[att_ind][0])
      # print(index_positions[att_ind][1]+1)
      attention_matx[index_positions[node_ind][0]:index_positions[node_ind][1]+1,index_positions[att_ind][0]:index_positions[att_ind][1]+1]=1
      # # print("node:",node_ind)
      # # print("attending to node:", att_ind)
      # # print("node pos: ",index_positions[node_ind][0],index_positions[node_ind][1]+1)
      # # print("attending node pos: ",header_numbering[att_ind][0],header_numbering[att_ind][1]+1)
      # global_attention_matx[index_positions[node_ind][0]:index_positions[node_ind][1]+1,
      # header_numbering[att_ind][0]:header_numbering[att_ind][1]+1]=1

  # print([list(x) for x in attention_matx])

  # for node_ind in header_pos:
  #   header_positions.append(index_positions[node_ind][0])
  #   global_attention_mask[index_positions[node_id][0]:index_positions[node_id][1]+1]=1
  
  for masked_id in masked_ids:
    masked_node_positions.append(index_positions[masked_id][0])
    # # print("masked node: ",masked_id)
    # # print("attend to all headers till ",no_header_tokens)
    # global_attention_matx[index_positions[masked_id][0]:index_positions[masked_id][1]+1,:no_header_tokens]=1

  # print("global_attention_mask: ",list(enumerate(global_attention_mask.tolist())))
  # print("global_attention_matx: ",global_attention_matx.tolist())

  # print("here")
  # print(header_positions)
  # print(masked_node_positions)
  # print(header_pos, masked_ids)
  masked_header_weights[:len(header_positions),:len(masked_node_positions)]=1
  
  for masked_pos,masked_id in enumerate(masked_ids):
    for head_pos,head_id in enumerate(header_pos):
      if head_id==masked_parent_ids[masked_pos]:
        # if parent set label to 1
        try:
          # print(masked_pos, masked_id, head_pos, head_id)
          masked_header_labels[head_pos,masked_pos]=1
        except Exception as e:
          print(e)
          # print("no headers=",no_headers)
          print(header_pos)
          for tok_list in token_list:
            print(json.dumps(tok_list))
          # print(json.dumps(doc,indent="\t"))
          exit()

      elif head_id in ancestor_dict[masked_id]:
        # remove ancestor which is not parent
        masked_header_weights[head_pos,masked_pos]=0
        masked_header_labels[head_pos, masked_pos] = -100
    
    # masked_header_weights[index_positions[ancestor_dict[masked_id]][0],index_positions[masked_id][0]]=0
    # # add parent
    # print(index_positions[masked_edges[masked_id]][0])
    # masked_header_weights[index_positions[masked_edges[masked_id]][0],index_positions[masked_id][0]]=1
    # masked_header_labels[index_positions[masked_edges[masked_id]][0],index_positions[masked_id][0]]=1

    # attend to all headers if masked
    # for h in header_pos:
      # print("attention:")
      # print("masked_id=",masked_id)
      # print("head_id=",head_id)
      # print(index_positions[masked_id][0])
      # print(index_positions[masked_id][1]+1)
      # print(index_positions[head_id][0])
      # print(index_positions[head_id][1]+1)
      attention_matx[index_positions[masked_id][0]:index_positions[masked_id][1]+1,index_positions[head_id][0]:index_positions[head_id][1]+1]=1
  

  # print(masked_header_weights)
  # print(masked_header_labels)

  while len(header_positions)<max_num_headers:
    header_positions.append(0)

  while len(masked_node_positions)<max_masked_nodes:
    masked_node_positions.append(0)

  # while len(tokens)<max_seq_length:
  #   tokens.append("0") # ?
  #   formats.append((0,0)) # ?
  #   bboxes.append((0,0,0,0))
  #   segment_ids.append(len(tokens_index_list)-1)

  # print(header_positions)
  # print(masked_node_positions)
  # print(masked_header_weights)
  # print(masked_header_labels)

  # print("before calling mlm:")
  # print(tokens)
  # print(segment_ids)
  (tokens, masked_lm_positions,masked_lm_labels) = create_masked_lm_predictions(
    tokens, masked_lm_prob, max_predictions_per_seq, tokenizer, rng)

  # print(tokens)

  # masked_header_weights=list(np.reshape(masked_header_weights,(-1,)))
  # masked_header_labels=list(np.reshape(masked_header_labels,(-1,)))

  # print(masked_header_weights)
  # print(masked_header_labels)

  # attention_matx=list(np.reshape(attention_matx,(-1,)))
  # local_attention_matx=list(np.reshape(local_attention_matx,(-1,)))
  # global_attention_matx=list(np.reshape(global_attention_matx,(-1,)))
  # global_attention_mask=list(np.reshape(global_attention_mask,(-1,)))
  attention_mask = np.ones((len(tokens),), dtype=int)
  # attention_mask[masked_lm_positions] = 0

  global_attention_mask = global_attention_mask*attention_mask
  assert len_tokens == len(tokens)
  masked_lm_ids = -100*np.ones((len_tokens, ))
  masked_lm_ids[masked_lm_positions] = masked_lm_labels


  global_attention_matx = attention_matx[global_attention_mask.astype(bool), :].T


  final_lam = np.zeros((attention_matx.shape[0], max_num_headers*max_tokens_per_header+attention_window+1))
  final_lam[:, :np.sum(global_attention_mask)] = attention_matx[:, global_attention_mask.astype(bool)]
  attention_matx = np.pad(attention_matx, ((0, 0), (attention_window//2, attention_window//2)), constant_values=0)

  for li in range(attention_matx.shape[0]):
      final_lam[li, -(attention_window+1):] = attention_matx[li, li:li+attention_window+1]
  # print(masked_header_labels)
  instance = TrainingInstance(
      tokens=tokens,#unpadded
      formats=formats,#unpadded
      bboxes=bboxes,#unpadded
      segment_ids=segment_ids,#unpadded
      attention_mask=attention_mask,#unpadded
      masked_lm_labels=masked_lm_ids,#unpadded
      masked_header_weights=masked_header_weights,
      masked_header_labels=masked_header_labels.reshape([-1]),
      local_attention_matx=final_lam,
      global_attention_mask=global_attention_mask,
      global_attention_matx=global_attention_matx,
      # attention_matx=attention_matx,
      header_positions=header_positions,
      masked_node_positions=masked_node_positions)

  return instance


MaskedLmInstance = collections.namedtuple("MaskedLmInstance",
                                          ["index", "label"])

# def create_masked_header_alignment(masked_node_prob,rng,forest,curr_node_for_index,max_masked_nodes):
#   if len(forest)==max_masked_nodes:
#     return forest
#   curr_node=forest[curr_node_for_index]

#   remove_list=[]

#   for node in curr_node.get("sublevels",[]):
#     if len(forest)==max_masked_nodes:
#       break
#     if rng.random()<masked_node_prob:
#       forest.append(node.copy())
#       remove_list.append(node.copy())
#       forest=create_masked_header_alignment(masked_node_prob,rng,forest,len(forest)-1,max_masked_nodes)

#   for node in remove_list:
#     forest[curr_node_for_index]["sublevels"].remove(node)

#   return forest

def is_empty(curr_node):
  if not curr_node.get("title",[]) and not curr_node.get("content",[]) and not curr_node.get("sublevels",[]):
    return True
  else:
    return False

def create_nodes_list(curr_node,attention_dict,max_seq_length,
  token_list,format_list,bbox_list,# the three lists are list of lists to be flattened later
  no_toks,ancestor_ind,header_pos, ancestors_list, ancestor_dict,
  masked_edges,masked_node_prob,max_masked_nodes,no_masks, samples,tokenizer,rng):
  """parse the generated tree and generate tokens etc
  ancestor_ind: list of ancestors that should be stored for the node
  attention_dict: node->ancestor_ind, node->non-masked children
  ancestors_list: list of actual ancestors of the node
  ancestor_dict: node->actual ancestor list
  header_pos: positions of header node in token_list
  masked_edges: masked child position->it's parent's position"""


  curr_node_toks=[]
  curr_node_formats=[]
  curr_node_bbox=[]

  if no_toks<max_seq_length:
    curr_node_toks.append(tokenizer.convert_tokens_to_ids(["</s>"])[0])#appending [NODE] token id in vocab
    curr_node_formats.append([0,0,0]) # ?
    curr_node_bbox.append([0,0,0,0])
    no_toks+=1

  else:
    # forget this node entirely

    return token_list,format_list,bbox_list,-1,attention_dict,no_toks,no_masks,samples, masked_edges,header_pos,ancestor_dict

  # sub_node_index_list=list(range(len(curr_node["sublevels"][0])))
  # rng.shuffle(sub_node_index_list)

  for token_ind in range(len(curr_node.get("title",[[]])[0])):
    if no_toks>=max_seq_length:
      break
    curr_node_toks.append(curr_node["title"][0][token_ind])
    curr_node_formats.append(curr_node["title"][1][token_ind])
    curr_node_bbox.append(curr_node["title"][2][token_ind])
    no_toks+=1


  token_list.append(curr_node_toks)
  format_list.append(curr_node_formats)
  bbox_list.append(curr_node_bbox)

  # title for current node added

  this_ind=len(token_list)-1

  ancestor_dict[this_ind]=ancestors_list.copy()

  header_pos.append(this_ind)

  attention_dict[this_ind]=ancestor_ind.copy() # this node should attend to all parents
  
  for cont in curr_node.get("content",[]):
    cont_toks_list=[]
    cont_font_list=[]
    cont_bbox_list=[]
    if no_toks<max_seq_length:
      cont_toks_list.append(tokenizer.convert_tokens_to_ids(["</s>"])[0])
      cont_font_list.append([0,0,0]) #?
      cont_bbox_list.append([0,0,0,0])
      no_toks+=1

      for token_ind in range(len(cont[0])):
        if no_toks<max_seq_length:
          cont_toks_list.append(cont[0][token_ind])
          cont_font_list.append(cont[1][token_ind]) #?
          cont_bbox_list.append(cont[2][token_ind])
          no_toks+=1
        else:
          break

      token_list.append(cont_toks_list)
      format_list.append(cont_font_list)
      bbox_list.append(cont_bbox_list)
      cont_ind=len(token_list)-1

      ancestor_dict[cont_ind]=ancestors_list+[this_ind]
      
      prob = rng.random()
      if no_masks<max_masked_nodes and prob<masked_node_prob:
        # mask this edge
        masked_edges[cont_ind]=this_ind
        attention_dict[cont_ind]=[] # since it is masked and has no child, attention is null
        no_masks+=1
      else:
        samples.append([cont_ind, this_ind, "L"])
        attention_dict[this_ind].append(cont_ind) # attend to child
        attention_dict[cont_ind]=ancestor_ind+[this_ind] # attend to all ancestors


  # ancestors_list.append(this_ind) # ancestor for children
  for node in curr_node.get("sublevels",[]):
    if is_empty(node):
      continue
    if no_masks<max_masked_nodes and rng.random()<masked_node_prob:
      # mask this edge
      # empty ancestor_ind
      token_list,format_list,bbox_list,child_node_ind,attention_dict,no_toks,no_masks, samples, masked_edges,header_pos,ancestor_dict=create_nodes_list(
        node,attention_dict,max_seq_length,token_list,format_list,bbox_list,no_toks,[],header_pos,ancestors_list+[this_ind], ancestor_dict,
        masked_edges,masked_node_prob,max_masked_nodes,no_masks+1, samples, tokenizer,rng)
      #print("here ", child_node_ind, no_masks)
      if child_node_ind!=-1:# otherwise child node was ditched
        masked_edges[child_node_ind]=this_ind
        #attention_dict[cont_ind]=[]#tarun?because it has child
        #no_masks+=1#tarun?
      else:
        no_masks -= 1
    else:
      token_list,format_list,bbox_list,child_node_ind,attention_dict,no_toks,no_masks, samples, masked_edges,header_pos,ancestor_dict=create_nodes_list(
        node,attention_dict,max_seq_length,token_list,format_list,bbox_list,no_toks,ancestor_ind+[this_ind],header_pos,ancestors_list+[this_ind], ancestor_dict,
        masked_edges,masked_node_prob,max_masked_nodes,no_masks,samples, tokenizer,rng)#no_mask+1 removed tarun
      #print("here -- ", child_node_ind, this_ind, no_masks)
      if child_node_ind!=-1:# otherwise child node was ditched
        samples.append([child_node_ind, this_ind, "S"])#tarun
        attention_dict[this_ind].append(child_node_ind)
        

  # ancestors_list=ancestors_list[:-1]

  return token_list,format_list,bbox_list,this_ind,attention_dict,no_toks,no_masks,samples, masked_edges,header_pos,ancestor_dict




def create_masked_lm_predictions(tokens, masked_lm_prob,
                                 max_predictions_per_seq, tokenizer, rng, do_whole_word_mask=False):
  """Creates the predictions for the masked LM objective."""

  cand_indexes = []
  
  for (i, token) in enumerate(tokens):
    if token==tokenizer.convert_tokens_to_ids(["</s>"])[0]:
      continue
    # Whole Word Masking means that if we mask all of the wordpieces
    # corresponding to an original word. When a word has been split into
    # WordPieces, the first token does not have any marker and any subsequence
    # tokens are prefixed with ##. So whenever we see the ## token, we
    # append it to the previous set of word indexes.
    #
    # Note that Whole Word Masking does *not* change the training code
    # at all -- we still predict each WordPiece independently, softmaxed
    # over the entire vocabulary.
    # if token == tokenizer.convert_tokens_to_ids(["."])[0]: tarun
    #   continue
    if (do_whole_word_mask and len(cand_indexes) >= 1 and#this is false here
        tokenizer.inv_vocab[token].startswith("##")):
      cand_indexes[-1].append(i)
    else:
      cand_indexes.append([i])

  rng.shuffle(cand_indexes)

  output_tokens = list(tokens)

  num_to_predict = min(max_predictions_per_seq,
                       max(1, int(round(len(tokens) * masked_lm_prob))))

  masked_lms = []
  covered_indexes = set()
  # print(len(cand_indexes))
  # print(cand_indexes,"****")
  for index_set in cand_indexes:
    if len(masked_lms) >= num_to_predict:
      break
    # If adding a whole-word mask would exceed the maximum number of
    # predictions, then just skip this candidate.
    if len(masked_lms) + len(index_set) > num_to_predict:
      continue
    is_any_index_covered = False
    for index in index_set:
      if index in covered_indexes:
        is_any_index_covered = True
        break
    if is_any_index_covered:
      continue
    for index in index_set:
      covered_indexes.add(index)

      masked_token = None
      # 80% of the time, replace with [MASK]
      if rng.random() < 0.8:
        masked_token = tokenizer.convert_tokens_to_ids(["<mask>"])[0]#tokenizer.vocab["[MASK]"]#in lonformer use tokenizer.convert_tokens_to_ids(["<mask>"])[0]
      else:
        # 10% of the time, keep original
        if rng.random() < 0.5:
          masked_token = tokens[index]
        # 10% of the time, replace with random word
        else:
          #masked_token = rng.randint(0, tokenizer.max_token)
          masked_token = rng.randint(0, len(tokenizer)-1)

      output_tokens[index] = masked_token

      masked_lms.append(MaskedLmInstance(index=index, label=tokens[index]))
  assert len(masked_lms) <= num_to_predict
  masked_lms = sorted(masked_lms, key=lambda x: x.index)

  masked_lm_positions = []
  masked_lm_labels = []
  for p in masked_lms:
    masked_lm_positions.append(p.index)
    masked_lm_labels.append(p.label)

  return (output_tokens, masked_lm_positions, masked_lm_labels)


def truncate_seq_pair(tokens_a, tokens_b, max_num_tokens, rng):
  """Truncates a pair of sequences to a maximum sequence length."""
  while True:
    total_length = len(tokens_a) + len(tokens_b)
    if total_length <= max_num_tokens:
      break

    trunc_tokens = tokens_a if len(tokens_a) > len(tokens_b) else tokens_b
    assert len(trunc_tokens) >= 1

    # We want to sometimes truncate from the front and sometimes from the
    # back to add more randomness and avoid biases.
    if rng.random() < 0.5:
      del trunc_tokens[0]
    else:
      trunc_tokens.pop()

########### for end task ###########

def create_training_data_from_instance_end_task(doc,attention_window,max_seq_length,masked_node_prob,max_masked_nodes,masked_lm_prob,
  max_num_headers,max_tokens_per_header,max_predictions_per_seq,tokenizer,rng):

  token_list=[]
  format_list=[]
  bbox_list=[]
  no_toks=0
  token_list,format_list,bbox_list,_,attention_dict,no_toks,no_masks, samples,masked_edges,header_pos,ancestor_dict=create_nodes_list(
    doc,{},max_seq_length,token_list,format_list,bbox_list,no_toks,[],[],[],{},
    {},masked_node_prob,max_masked_nodes,0,[],tokenizer,rng)

  if no_masks == 0:
    #print("samples", samples)
    num_mask = min(rng.randint(1, 3), len(samples))
    mask_edges_sample = rng.sample(samples, num_mask)
    
    for each_edge in mask_edges_sample:
      masked_edges[each_edge[0]]=each_edge[1]
      attention_dict[each_edge[1]].remove(each_edge[0])
      if each_edge[-1] == "L":#if leaf
        attention_dict[each_edge[0]]=[]
      no_masks += 1
    
  masked_ids=[]
  masked_parent_ids=[]
  for masked_id in masked_edges:
    masked_ids.append(masked_id)
    masked_parent_ids.append(masked_edges[masked_id])

  tokens_index_list=list(range(len(token_list))) # shuffle obtained nodes

  # rng.shuffle(tokens_index_list)
  len_tokens = sum(map(len, token_list))
  # local_attention_matx=np.zeros((len_tokens,len_tokens)) # attention local to the node

  # global_attention_matx=np.zeros((len_tokens,max_num_headers*max_tokens_per_header)) # attention for header nodes

  global_attention_mask=np.zeros((len_tokens,), dtype=int)
  attention_matx=np.zeros((len_tokens,len_tokens))


  tokens=[]
  formats=[]
  bboxes=[]
  segment_ids=[]
  header_positions=[]
  masked_node_positions=[]

  masked_header_labels=np.zeros((max_num_headers,max_masked_nodes),dtype=int)
  masked_header_weights=np.zeros((max_num_headers,max_masked_nodes))

  index_positions=np.zeros((len(token_list),2),dtype=int)

  header_numbering={}
  no_header_tokens=0

  for s_id,node_ind in enumerate(tokens_index_list):
    start_pos=len(tokens)
    index_positions[node_ind][0]=start_pos
    for tok_ind in range(len(token_list[node_ind])):
      tokens.append(token_list[node_ind][tok_ind])
      formats.append(format_list[node_ind][tok_ind])
      bboxes.append(bbox_list[node_ind][tok_ind])
      segment_ids.append(s_id)
    end_pos=len(tokens)-1
    index_positions[node_ind][1]=end_pos

    assert(end_pos+1>start_pos)

    if node_ind in header_pos:
      header_positions.append(start_pos)
      global_attention_mask[start_pos:end_pos+1]=1
      # print("global_attention_mask sublist:",global_attention_mask[start_pos:end_pos+1])
      header_numbering[node_ind]=[no_header_tokens,no_header_tokens+end_pos-start_pos]
      # print("header_numbering: ",header_numbering[node_ind])
      no_header_tokens+=1+end_pos-start_pos

    # self-attention
    # local_attention_matx[start_pos:end_pos+1,start_pos:end_pos+1]=1
    attention_matx[start_pos:end_pos+1,start_pos:end_pos+1]=1

  # attention by tree structure
  # restrict to ancestors
  for node_ind in tokens_index_list:
    for att_ind in attention_dict.get(node_ind,[]):
     
      attention_matx[index_positions[node_ind][0]:index_positions[node_ind][1]+1,index_positions[att_ind][0]:index_positions[att_ind][1]+1]=1
  
  for masked_id in masked_ids:
    masked_node_positions.append(index_positions[masked_id][0])
    
  masked_header_weights[:len(header_positions),:len(masked_node_positions)]=1
  
  for masked_pos,masked_id in enumerate(masked_ids):
    for head_pos,head_id in enumerate(header_pos):
      if head_id==masked_parent_ids[masked_pos]:
        # if parent set label to 1
        try:
          # print(masked_pos, masked_id, head_pos, head_id)
          masked_header_labels[head_pos,masked_pos]=1
        except Exception as e:
          print(e)
          # print("no headers=",no_headers)
          print(header_pos)
          for tok_list in token_list:
            print(json.dumps(tok_list))
          # print(json.dumps(doc,indent="\t"))
          exit()

      elif head_id in ancestor_dict[masked_id]:
        # remove ancestor which is not parent
        masked_header_weights[head_pos,masked_pos]=0
        masked_header_labels[head_pos, masked_pos] = -100

      attention_matx[index_positions[masked_id][0]:index_positions[masked_id][1]+1,index_positions[head_id][0]:index_positions[head_id][1]+1]=1
  

  # print(masked_header_weights)
  # print(masked_header_labels)

  while len(header_positions)<max_num_headers:
    header_positions.append(0)

  while len(masked_node_positions)<max_masked_nodes:
    masked_node_positions.append(0)

  # (tokens, masked_lm_positions,masked_lm_labels) = create_masked_lm_predictions(
  #   tokens, masked_lm_prob, max_predictions_per_seq, tokenizer, rng)

  attention_mask = np.ones((len(tokens),), dtype=int)
  # attention_mask[masked_lm_positions] = 0

  global_attention_mask = global_attention_mask*attention_mask
  assert len_tokens == len(tokens)
  # masked_lm_ids = -100*np.ones((len_tokens, ))
  # masked_lm_ids[masked_lm_positions] = masked_lm_labels


  global_attention_matx = attention_matx[global_attention_mask.astype(bool), :].T


  final_lam = np.zeros((attention_matx.shape[0], max_num_headers*max_tokens_per_header+attention_window+1))
  final_lam[:, :np.sum(global_attention_mask)] = attention_matx[:, global_attention_mask.astype(bool)]
  attention_matx = np.pad(attention_matx, ((0, 0), (attention_window//2, attention_window//2)), constant_values=0)

  for li in range(attention_matx.shape[0]):
      final_lam[li, -(attention_window+1):] = attention_matx[li, li:li+attention_window+1]
  # print(masked_header_labels)
  instance = TrainingInstanceForEndTask(
      tokens=tokens,#unpadded
      formats=formats,#unpadded
      bboxes=bboxes,#unpadded
      segment_ids=segment_ids,#unpadded
      attention_mask=attention_mask,#unpadded
      local_attention_matx=final_lam,
      global_attention_mask=global_attention_mask,
      global_attention_matx=global_attention_matx)

  return instance

####################################
def main():

  tokenizer = tokenization.FullTokenizer(
      vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

  input_files = FLAGS.input_file.split(",")

  print("*** Reading from input files ***")

  rng = random.Random(FLAGS.random_seed)
  create_training_instances(
      input_files, tokenizer, FLAGS.max_seq_length, FLAGS.dupe_factor,
      FLAGS.short_seq_length, FLAGS.masked_lm_prob, FLAGS.deep_tree_prob,
      FLAGS.masked_node_prob,FLAGS.max_num_headers,FLAGS.max_tokens_per_header,FLAGS.max_masked_nodes,
      # FLAGS.masked_node_prob,FLAGS.max_num_headers,FLAGS.max_masked_nodes,
      FLAGS.max_predictions_per_seq, rng, FLAGS.output_dir, FLAGS.written_offset)

  # output_files = FLAGS.output_file.split(",")
  # print("*** Writing to output files ***")
  # for output_file in output_files:
  #   print("  %s", output_file)

  # write_instance_to_example_files(instances, tokenizer, FLAGS.max_seq_length,
  #                                 FLAGS.max_predictions_per_seq, FLAGS.output_dir, FLAGS.written_offset)


if __name__ == "__main__":

  parser = argparse.ArgumentParser(prog='create_pretraining_data',
                                   usage='%(prog)s [options]',)


  parser.add_argument("--input_file", default=None,
                      help="Input raw text file (or comma-separated list of files).")

  parser.add_argument(
      "--output_dir", default=None,
      help="Output TF example file (or comma-separated list of files).")

  parser.add_argument("--vocab_file", default=None,
                      help="The vocabulary file that the BERT model was trained on.")

  parser.add_argument(
      "--written_offset", default=None,
      help="Instances already written to disk")


  parser.add_argument(
      "--do_lower_case", default=True, action="store_true",
      help="Whether to lower case the input text. Should be True for uncased "
      "models and False for cased models.")

  parser.add_argument(
      "--do_whole_word_mask", default=False, action="store_true",
      help="Whether to use whole word masking rather than per-WordPiece masking.")
  do_whole_word_mask = False

  parser.add_argument("--max_seq_length", default=512, help="Maximum sequence length.", type=int)

  parser.add_argument("--max_predictions_per_seq", default=20,
                       help="Maximum number of masked LM predictions per sequence.", type=int)

  parser.add_argument(
    "--max_num_headers",default=15, # avg no headers=9 with sd=5
    help="Maximum number of headers that will be present in the input", type=int)


  parser.add_argument(
    "--max_tokens_per_header",default=15, # average 5 words per header, sd=2. multiply by 2 for tokenisation
    help="Maximum number of tokens expected per header", type=int)

  parser.add_argument(
    "--max_masked_nodes",default=5, # what should this number be?
    help="Maximum number of nodes in the tree whose parent will be masked", type=int)

  parser.add_argument("--random_seed", default=12345, help="Random seed for data generation.", type=int)

  parser.add_argument(
      "--dupe_factor", default=10,
      help="Number of times to duplicate the input data (with different masks).", type=int)

  parser.add_argument("--masked_lm_prob", default=0.15, help="Masked LM probability.", type=float)

  parser.add_argument("--deep_tree_prob",default=0.5,
    help="Probability with which to create deep tree instance from doc", type=float)

  parser.add_argument("--short_seq_length",default=30,
    help="max no of tokens per node in bottom up creation", type=int)

  parser.add_argument("--masked_node_prob",default=0.1, # avg no paras/header=4, so avg no paras around 30. total average 30+8+.. around 40-45 edges.
    # set this so that at an average, even the last edge has a decent prob of being masked and 5 edges are masked overall with good prob
    # setting to 0.1 so that only 4 are masked till the 39th edge with prob 0.02 so that with arnd 0.03 prob at most 4 are masked overall
    help="Probability with which an edge will be masked", type=float)

  # parser.add_argument(
  #     "short_seq_prob", default=0.1,
  #     "Probability of creating sequences which are shorter than the "
  #     "maximum length.", type=float)

  parser.add_argument("--window_size",default=512,
    help="Window size for local attention of longformer", type=int)

  FLAGS = parser.parse_args()

  window_size=FLAGS.window_size

  if not FLAGS.input_file or not FLAGS.output_dir or not FLAGS.vocab_file or not FLAGS.written_offset:
    raise Exception("input_file, output_dir, written_offset & vocab_file are required")

  main()

