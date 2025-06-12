import torch
import torch.nn as nn
from torch.nn import Parameter, init
from torch import Tensor
import math
import re


def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=-100):
    """From fairseq"""
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.0)
        smooth_loss.masked_fill_(pad_mask, 0.0)
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
    if ignore_index is not None:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)

    nll_loss = nll_loss.sum()
    smooth_loss = smooth_loss.sum()
    eps_i = epsilon / lprobs.size(-1)
    loss = (1.0 - epsilon) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss


def shift_tokens_left(input_ids: torch.Tensor, pad_token_id: int):
    """
    Shift input ids one token to the right.
    """
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, :-1] = input_ids[:, 1:].clone()
    shifted_input_ids[:, -1] = pad_token_id
    assert pad_token_id is not None, "self.model.config.pad_token_id has to be defined."
    return shifted_input_ids


def extract_vietnamese_legal_triplets(text):
    """
    Extract triplets from Vietnamese legal text in format:
    <ENTITY_TYPE> Entity_Text <ENTITY_TYPE> Entity_Text <RELATION_TYPE>
    """
    triplets = []
    text = text.strip()
    
    # Remove special tokens
    text = text.replace("<s>", "").replace("<pad>", "").replace("</s>", "")
    
    # Define entity types for Vietnamese legal domain
    entity_types = [
        "<ORGANIZATION>", "<LOCATION>", "<DATE/TIME>", "<LEGAL_PROVISION>",
        "<RIGHT/DUTY>", "<PERSON>", "<Effective_From>", "<Applicable_In>",
        "<Relates_To>", "<Amended_By>"
    ]
    
    # Split text into tokens
    tokens = text.split()
    
    i = 0
    while i < len(tokens):
        # Look for entity type tokens
        if tokens[i] in entity_types:
            head_type = tokens[i]
            head_text = ""
            i += 1
            
            # Extract head entity text
            while i < len(tokens) and tokens[i] not in entity_types:
                head_text += " " + tokens[i]
                i += 1
            
            # Look for tail entity type
            if i < len(tokens) and tokens[i] in entity_types:
                tail_type = tokens[i]
                tail_text = ""
                i += 1
                
                # Extract tail entity text
                while i < len(tokens) and tokens[i] not in entity_types:
                    tail_text += " " + tokens[i]
                    i += 1
                
                # Look for relation type
                if i < len(tokens) and tokens[i] in entity_types:
                    relation_type = tokens[i]
                    
                    # Add triplet
                    if head_text.strip() and tail_text.strip():
                        triplets.append({
                            'head': head_text.strip(),
                            'head_type': head_type,
                            'tail': tail_text.strip(),
                            'tail_type': tail_type,
                            'type': relation_type
                        })
                    i += 1
                else:
                    i += 1
            else:
                i += 1
        else:
            i += 1
    
    return triplets


def extract_triplets_typed(text, mapping_types=None):
    """
    Fallback to original REBEL format if needed
    """
    if mapping_types is None:
        mapping_types = {
            '<ORGANIZATION>': 'ORGANIZATION', 
            '<LOCATION>': 'LOCATION', 
            '<PERSON>': 'PERSON',
            '<LEGAL_PROVISION>': 'LEGAL_PROVISION',
            '<DATE/TIME>': 'DATE/TIME',
            '<RIGHT/DUTY>': 'RIGHT/DUTY'
        }
    
    triplets = []
    relation = ''
    text = text.strip()
    current = 'x'
    subject, relation, object_, object_type, subject_type = '','','','',''

    for token in text.replace("<s>", "").replace("<pad>", "").replace("</s>", "").split():
        if token == "<triplet>":
            current = 't'
            if relation != '':
                triplets.append({
                    'head': subject.strip(), 
                    'head_type': subject_type, 
                    'type': relation.strip(),
                    'tail': object_.strip(), 
                    'tail_type': object_type
                })
                relation = ''
            subject = ''
        elif token in mapping_types:
            if current == 't' or current == 'o':
                current = 's'
                if relation != '':
                    triplets.append({
                        'head': subject.strip(), 
                        'head_type': subject_type, 
                        'type': relation.strip(),
                        'tail': object_.strip(), 
                        'tail_type': object_type
                    })
                object_ = ''
                subject_type = mapping_types[token]
            else:
                current = 'o'
                object_type = mapping_types[token]
                relation = ''
        else:
            if current == 't':
                subject += ' ' + token
            elif current == 's':
                object_ += ' ' + token
            elif current == 'o':
                relation += ' ' + token
                
    if subject != '' and relation != '' and object_ != '' and object_type != '' and subject_type != '':
        triplets.append({
            'head': subject.strip(), 
            'head_type': subject_type, 
            'type': relation.strip(),
            'tail': object_.strip(), 
            'tail_type': object_type
        })
    return triplets


class BartTripletHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(
        self,
        input_dim: int,
        inner_dim: int,
        num_classes: int,
        pooler_dropout: float,
    ):
        super().__init__()
        self.dense_head_tail_ctxt = nn.Linear(input_dim*3, inner_dim)
        self.dropout = nn.Dropout(p=pooler_dropout)
        self.out_proj = nn.Linear(inner_dim, num_classes)

    def forward(self, head_states: torch.Tensor, tail_states: torch.Tensor, context_states: torch.Tensor):
        combined_state = torch.cat((head_states, tail_states, context_states), dim = 1)
        combined_state = self.dropout(combined_state)
        hidden_states = self.dense_head_tail_ctxt(combined_state)
        hidden_states = torch.tanh(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.out_proj(hidden_states)
        return hidden_states 