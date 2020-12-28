"""PyTorch BERT model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import copy
import json
import math
import logging
import tarfile
import tempfile
import shutil
import numpy as np


import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
import torch.nn.functional as F

import fastseq
from transformers.file_utils import cached_path
from transformers.generation_utils import GenerationMixin

from torch.nn.modules.loss import _Loss

from s2s_ft.modeling_decoding import (
        BertForSeq2SeqDecoder,
        PreTrainedBertModel, 
        BertConfig, 
        BertEmbeddings,
        BertSelfOutput,
        BertIntermediate,
        BertOutput,
        BertPooler,
        BertPreTrainingHeads, 
        BertAttention as OriBertAttention,
        BertLayer as OriBertLayer,
        BertModel as OriBertModel,
        BertEncoder as OriBertEncoder
)

def _reorder_buffer(attn_cache, beam_idx):
    for k, input_buffer_k in attn_cache.items():
        if input_buffer_k is not None and 'enc' not in k:
            attn_cache[k] = input_buffer_k.index_select(0, beam_idx)
    return attn_cache

class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super(BertSelfAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.output_attentions = config.output_attentions

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask, history_states=None,**kwargs):
        new_query_layer = self.query(hidden_states)
        new_key_layer = self.key(hidden_states)
        new_value_layer = self.value(hidden_states)

        prev_enc_key_layer = history_states['prev_enc_key_layer'] if history_states is not None and 'prev_enc_key_layer' in history_states else None
        prev_enc_value_layer = history_states['prev_enc_value_layer'] if history_states is not None and 'prev_enc_value_layer' in history_states else None
        prev_dec_key_layer = history_states['prev_dec_key_layer'] if history_states is not None and 'prev_dec_key_layer' in history_states else None
        prev_dec_value_layer = history_states['prev_dec_value_layer'] if history_states is not None and 'prev_dec_value_layer' in history_states else None

        query_layer = self.transpose_for_scores(new_query_layer)
        key_layer = self.transpose_for_scores(new_key_layer)
        value_layer = self.transpose_for_scores(new_value_layer)
        if prev_enc_key_layer is not None:
            enc_size = prev_enc_key_layer.size()
            enc_attention_scores = torch.einsum("bxhtd,bhsd->bxhts", query_layer.view(enc_size[0], -1, *query_layer.size()[1:]), prev_enc_key_layer)
            enc_attention_scores = enc_attention_scores.reshape(-1, *enc_attention_scores.size()[2:])
            if prev_dec_key_layer is not None:
                key_layer = torch.cat((prev_dec_key_layer, key_layer), dim=2)
                value_layer = torch.cat((prev_dec_value_layer, value_layer), dim=2)
            dec_attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
            enc_attention_scores = enc_attention_scores / math.sqrt(self.attention_head_size)
            dec_attention_scores = dec_attention_scores / math.sqrt(self.attention_head_size)
            attention_scores = torch.cat((enc_attention_scores, dec_attention_scores), dim=-1)
            attention_scores = attention_scores + attention_mask
            attention_probs = nn.Softmax(dim=-1)(attention_scores)
            attention_probs = self.dropout(attention_probs)

            enc_attention_probs = attention_probs[:, :, :, :enc_size[2]]
            dec_attention_probs = attention_probs[:, :, :, enc_size[2]:]
            enc_attention_probs = enc_attention_probs.to(prev_enc_value_layer.dtype)
            enc_context_layer = torch.einsum("bxhtd,bhds->bxhts", enc_attention_probs.view(enc_size[0], -1, *enc_attention_probs.size()[1:]), prev_enc_value_layer)
            enc_context_layer = enc_context_layer.reshape(-1, *enc_context_layer.size()[2:])
            dec_context_layer = torch.matmul(dec_attention_probs, value_layer)
            context_layer = enc_context_layer + dec_context_layer

        else:
            attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
            attention_scores = attention_scores / math.sqrt(self.attention_head_size)
            attention_scores = attention_scores + attention_mask

            attention_probs = nn.Softmax(dim=-1)(attention_scores)
            attention_probs = self.dropout(attention_probs)

            context_layer = torch.matmul(attention_probs, value_layer)

        if history_states is None or len(history_states) == 0:
            history_states.update(dict({
                'prev_enc_key_layer': key_layer,
                'prev_enc_value_layer': value_layer
            }))
        else:
            history_states.update(dict({
                'prev_enc_key_layer': prev_enc_key_layer,
                'prev_enc_value_layer': prev_enc_value_layer,
                'prev_dec_key_layer': key_layer[:, :, :-1, :],
                'prev_dec_value_layer': value_layer[:, :, :-1, :]
            }))

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        return context_layer

class BertAttention(OriBertAttention):
    def __init__(self, config):
        super(BertAttention, self).__init__(config)
        self.self = BertSelfAttention(config)

class BertLayer(OriBertLayer):
    def __init__(self, config):
        super(BertLayer, self).__init__(config)
        self.attention = BertAttention(config)

class BertEncoder(OriBertEncoder):
    def __init__(self, config):
        super(BertEncoder, self).__init__(config)
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])
    def forward(self, hidden_states, attention_mask, history_states=None,output_all_encoded_layers=True,**kwargs):
        all_hidden_states = ()
        all_attentions = ()

        for i, layer_module in enumerate(self.layer):
            layer_outputs = layer_module(hidden_states, attention_mask, history_states=history_states[i])
            
            if output_all_encoded_layers:
                all_hidden_states = all_hidden_states + (hidden_states,)

            hidden_states = layer_outputs#[0]

        # Add last layer
        #if not output_all_encoded_layers:
        all_hidden_states = all_hidden_states + (hidden_states, )

        return all_hidden_states
        #outputs = (hidden_states,)
        #if output_all_encoded_layers:
        #    outputs = outputs + (all_hidden_states,)
        #return outputs  # last-layer hidden state, (all hidden states), (all attentions), (all encoder layers)

class BertModel(OriBertModel):
    def __init__(self, config):
        super().__init__(config)
        self.encoder = BertEncoder(config)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids, position_ids, attention_mask, output_all_encoded_layers=True, history_states= None):
        extended_attention_mask = self.get_extended_attention_mask(
                input_ids, token_type_ids, attention_mask)
        if history_states is None:
            history_states = [dict().copy() for _ in range(self.config.num_hidden_layers)]
        embedding_output = self.embeddings(
            input_ids, token_type_ids, position_ids)
        encoded_layers = self.encoder(embedding_output,
                                      extended_attention_mask,
                                      output_all_encoded_layers=output_all_encoded_layers,
                                      history_states=history_states)
        sequence_output = encoded_layers[-1]
        pooled_output = self.pooler(sequence_output)
        if not output_all_encoded_layers:
            encoded_layers = encoded_layers[-1]
        return sequence_output,pooled_output, encoded_layers, history_states

class BertForSeq2SeqDecoderFast(BertForSeq2SeqDecoder,GenerationMixin):
    def __init__(self, config, mask_word_id=0, num_labels=2, num_rel=0,
                 search_beam_size=1, length_penalty=1.0, eos_id=0, sos_id=0,
                 forbid_duplicate_ngrams=False, forbid_ignore_set=None, ngram_size=3, min_len=0, mode="s2s",
                 pos_shift=False,max_len=0,pad_id=0):
        super().__init__(config, mask_word_id, num_labels, num_rel,
                 search_beam_size, length_penalty, eos_id, sos_id,
                 forbid_duplicate_ngrams, forbid_ignore_set, ngram_size, min_len, mode,
                 pos_shift)
        self.bert = BertModel(config)
        self.cls = BertPreTrainingHeads(
            config, self.bert.embeddings.word_embeddings.weight, num_labels=num_labels)
        self.apply(self.init_bert_weights)
        self.crit_mask_lm = nn.CrossEntropyLoss(reduction='none')
        self.crit_next_sent = nn.CrossEntropyLoss(ignore_index=-1)
        self.mask_word_id = mask_word_id
        self.num_labels = num_labels
        self.num_rel = num_rel
        if self.num_rel > 0:
            self.crit_pair_rel = BertPreTrainingPairRel(
                config, num_rel=num_rel)
        self.search_beam_size = search_beam_size
        self.length_penalty = length_penalty
        self.eos_id = eos_id
        self.sos_id = sos_id
        self.forbid_duplicate_ngrams = forbid_duplicate_ngrams
        self.forbid_ignore_set = forbid_ignore_set
        self.ngram_size = ngram_size
        self.min_len = min_len
        assert mode in ("s2s", "l2r")
        self.mode = mode
        self.pos_shift = pos_shift
        self.dec_max_seq_length = max_len
        self.pad_id = pad_id


    def forward(self, src_in, **kwargs):
        if self.training:
            token_ids, seg_ids, token_mask, pos_ids = src_in
            output = self.bert(input_ids=token_ids, token_type_ids=seg_ids, attention_mask=token_mask, position_ids=pos_ids)
            output = self.cls(output[0], output[1])
            return output[0]
        dec_token, token_mask, pos_ids = src_in
        dec_token = torch.cat([dec_token, self.dec_mask_token], 1)
        dec_len = dec_token.size(1)
        dec_token = dec_token[:, -2:]
        dec_mask = token_mask[:, dec_len-2:dec_len, :self.src_state['src_len']+dec_len]
        dec_pos = pos_ids[:, dec_len-2:dec_len]
        history_states = kwargs['history_states']

        outputs = self.bert(input_ids=dec_token, token_type_ids=self.dec_seg[:, dec_len-2:dec_len], attention_mask=dec_mask, position_ids = dec_pos, history_states=history_states)
        output, _ = self.cls(outputs[0], outputs[1])  # Pick the last step: (bh * bm) * d_h
        state4cache = [pos_ids, token_mask] + outputs[-1]
        return output, state4cache

    @staticmethod
    def _reorder_cache(past, beam_idx):
        pos_ids, token_mask, history_states = past[0], past[1], past[2:]
        reordered_past = []
        for layer_past in history_states:
            reordered_past.append(_reorder_buffer(layer_past, beam_idx))
        newpast = [pos_ids, token_mask] + reordered_past
        return newpast

    def prepare_inputs_for_generation(self, token_ids, past=None, **kwargs):
        if past is None:
            active_batch_size, _ = token_ids.size()
            src_token, src_seg, src_pos, src_mask = self.src_state['src_token'], self.src_state['src_seg'], self.src_state['src_pos'], self.src_state['src_mask']
            src_len = self.src_state['src_len']
            outputs = self.bert(input_ids=src_token[:, :src_len], token_type_ids=src_seg[:, :src_len], attention_mask=src_mask[:, :src_len, :src_len], position_ids=src_pos[:, :src_len])
            token_pos = src_pos.repeat(1, self.search_beam_size).view(active_batch_size, src_pos.size(1))
            token_pos = token_pos[:, src_len:]
            token_mask = src_mask.unsqueeze(1).repeat(1, self.search_beam_size, 1, 1).view(active_batch_size, src_mask.size(1), src_mask.size(1))
            token_mask = token_mask[:, src_len:, :]
            history_states = outputs[-1]
        else:
            token_pos, token_mask, history_states = past[0], past[1], past[2:]
        ret = dict({
            'src_in': (token_ids, token_mask, token_pos),
            'history_states': history_states
        })
        return ret
    def get_output_embeddings(self):
        return self.cls.predictions.decoder
    def beam_search(self, input_ids,token_type_ids,position_ids,attention_mask,task_idx=None,mask_qkv=None):
        self.src_state = dict({
            'src_len': input_ids.size(1),
            'src_token': input_ids,
            'src_seg': token_type_ids,
            'src_mask': attention_mask,
            'src_pos': position_ids
        })
        #print(position_ids.size(1))
        #print(input_ids.size(1))
        #print(attention_mask)
        batch_size = input_ids.size(0)
        position_ids -= torch.cat((torch.eq(input_ids,torch.ones_like(input_ids)*self.mask_word_id).int(),
                                   torch.ones([batch_size,position_ids.size(1)-input_ids.size(1)])),-1)
        position_ids = torch.cat((position_ids,position_ids[:,-1:]+1),-1)
        attention_mask[:,:,input_ids.size(1)] = torch.zeros_like(attention_mask[:,:,input_ids.size(1)])
        #print(position_ids)
        #print(attention_mask)
        dec_seg = [0] + [1] * (self.dec_max_seq_length)
        self.dec_seg = torch.tensor(dec_seg, dtype=torch.long, device=input_ids.device).unsqueeze(0).repeat(input_ids.size(0) * self.search_beam_size, 1)
        self.dec_mask_token = torch.from_numpy(np.array([self.mask_word_id])).repeat([input_ids.size(0) * self.search_beam_size]).unsqueeze(-1).to(input_ids.device)
        bos_token = torch.from_numpy(np.array([self.sos_id])).repeat([batch_size]).unsqueeze(-1)
        if torch.cuda.is_available():
            bos_token = bos_token.cuda()

        batch_hyp = self.generate(
            bos_token,
            max_length=self.dec_max_seq_length,
            min_length=self.min_len,
            do_sample=False,
            num_beams=self.search_beam_size,
            no_repeat_ngram_size=self.ngram_size,
            length_penalty=self.length_penalty,
            repetition_penalty=1.0, #TODO:#repetition_penalty,
            bad_words_ids=[[10]],#TODO:self.forbid_duplicate_ngrams,
            bos_token_id=self.sos_id,
            pad_token_id=self.pad_id,#TODO: self.,
            eos_token_id=self.eos_id,
            num_return_sequences=1, #TODO: add nrs
            early_stopping=False,
            use_cache=True,
            temperature=1.0,
            top_k=1,
            top_p=1,
            decoder_start_token_id=self.sos_id
        )

        batch_hyp = batch_hyp.cpu().numpy()
        #batch_hyp = batch_hyp.reshape(batch_size, dec_num_return_sequences, -1)
        batch_hyp = batch_hyp.reshape(batch_size,-1)[:,1:]
        #batch_hyp = [[list(filter(lambda x: x != self.pad_id, b)) for b in a] for a in batch_hyp]
        #batch_scores = [ list(range(len(a), 0, -1)) for a in batch_hyp]
        #print(batch_hyp)
        return {"pred_seq":batch_hyp}#,None#, "score":batch_scores},None

