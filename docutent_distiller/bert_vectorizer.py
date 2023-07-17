import json
import os
from typing import List, Tuple, Union

import matplotlib.pylab as plt
import numpy as np
import torch
from transformers import BertModel, BertTokenizer
from tqdm import tqdm


class BertVectorizerCLS:
    def __init__(self):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.tokenizer = BertTokenizer.from_pretrained("SZTAKI-HLT/hubert-base-cc")
        self.model = BertModel.from_pretrained(
            "SZTAKI-HLT/hubert-base-cc",
            output_hidden_states=True,
            # Whether the model returns all hidden-states.
        )
        self.model = self.model.to(self.device)

    def get_tokens_number(self, text: str, return_tokens=False) -> Union[int, Tuple[int, list]]:

        tokenized_input = self.tokenizer.tokenize(text)
        if return_tokens:
            tokens, readables = ([], [])
            tokens.append(self.tokenizer.encode(text))
            for token in tokens:
                readables.append(self.tokenizer.convert_ids_to_tokens(token))
            return len(tokenized_input), readables
        else:
            return len(tokenized_input)

    def get_cls_token_embedding(self, list_of_texts: List[str]):
        self.model.eval()
        # tokenized_simple = self.tokenizer.encode_plus(list_of_texts, add_special_tokens=True, truncation=True, max_length=512)

        tokenizer_output = self.tokenizer(list_of_texts, add_special_tokens=True, truncation=True, padding=True)

        padded = tokenizer_output['input_ids']
        attention_mask = tokenizer_output['attention_mask']

        input_ids = torch.tensor(padded).to(self.device)
        attention_mask = torch.tensor(attention_mask).to(self.device)

        with torch.no_grad():
            last_hidden_states = self.model(input_ids, attention_mask=attention_mask)
            features = last_hidden_states[0][:, 0, :].cpu().numpy()
            return features

    def get_encoding_dict(self, input_text: str):
        """
        Returns <class transformers.tokenization_utils_base.BatchEncoding> with the following fields:
          - input_ids: list of token ids
          - token_type_ids: list of token type ids
          - attention_mask: list of indices (0,1) specifying which tokens should be considered by the model (return_attention_mask = True).
        """
        return self.tokenizer.encode_plus(
            input_text,
            add_special_tokens=True,
            max_length=256,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
