from typing import List

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel



class BertLongVectorizer:

    def __init__(self, model_name = "SZTAKI-HLT/hubert-base-cc"):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model = self.model.to(self.device)
        self.max_segment_length = self.model.config.max_position_embeddings - 2
        self.slices = []
        self.embedddings = None
        self.tokens = None
        self.connected_sw_tokens = None
        self.slicing_points = None

    def vectorize(self, text:str, matrix=True) -> np.ndarray:
        """
        Creates embeddings with huBERT model. If :param text: is longer than 510 subword level tokens, slices it into
        max. 510 subword-level token pieces (at word limits), then creates CLS token embeddings to these slices.

        :param matrix: True: returns all CLS vectors for slices, False: returns mean (axis=0) of CLS vectors
        :param text: Text for embedding
        :return: Embeddings from huBERT
        """
        if text == "":
            return np.zeros(1, )

        self.generate_matrix = matrix
        self.tokens = self.tokenizer.encode_plus(text,
                                            add_special_tokens=False)  # self.tokeinzer(text, add_special_tokens=False)

        word_ends = self._search_word_ends()
        self.slicing_points = self._search_slicing_points(word_ends)

        previous = 0
        slices = []
        for position in self.slicing_points:
            slices.append(self.tokens['input_ids'][previous:position+1])
            previous = position + 1

        self.slices = slices
        self.embedddings = self._create_embeddings()
        return self.embedddings

    def _search_word_ends(self) -> List[int]:

        self.connected_sw_tokens = self.tokens.word_ids()
        word_ends = []

        for index, sw in enumerate(self.connected_sw_tokens):
            if index == 0:
                continue
            if not self.connected_sw_tokens[index] == self.connected_sw_tokens[index - 1]:
                word_ends.append(index - 1)
            if index == len(self.connected_sw_tokens)-1:
                word_ends.append(index)

        return word_ends


    def _create_embeddings(self):
        input_ids, attention_masks = ([] for i in range(2))

        for slice in self.slices:
            input_ids_chunk = [2] + slice + [3]
            attention_mask_chunk = [1] * len(input_ids_chunk)

            input_ids_chunk += [0] * (self.max_segment_length - len(input_ids_chunk) + 2)
            attention_mask_chunk += [0] * (self.max_segment_length - len(attention_mask_chunk) + 2)
            input_ids.append(input_ids_chunk)
            attention_masks.append(attention_mask_chunk)

        input_ids = torch.tensor(input_ids).to(self.device)
        attention_mask = torch.tensor(attention_masks).to(self.device)

        # input_ids: egy tensor-ban a slice-ok input id-i (attention_mask szintén! )
        with torch.no_grad():
            last_hidden_states = self.model(input_ids, attention_mask=attention_mask)
            features = last_hidden_states[0][:, 0, :].cpu().numpy()
            features = np.squeeze(features)

        if self.generate_matrix:
            return features
        if features.shape == (768, ):
            return features
        return features.mean(axis=0)

    def _search_slicing_points(self, word_end_indices: List[int]) -> List[int]:
        slice_end_positions = []
        backtrack_from = self.max_segment_length
        backtrack_to = 0

        if word_end_indices[-1] >= self.max_segment_length:
            while backtrack_from <= word_end_indices[-1]:
                for possible_slice in reversed(range(backtrack_to, backtrack_from)):
                    if possible_slice in word_end_indices:
                        # if (backtrack_from + self.max_segment_length) <= word_end_indices[-1]:
                        slice_end_positions.append(possible_slice)
                        backtrack_to = backtrack_from + 1
                        backtrack_from = possible_slice + self.max_segment_length
                        break

        slice_end_positions.append(word_end_indices[-1])

        return slice_end_positions

if __name__ == '__main__':
    vectorizer = BertLongVectorizer()
    vectorizer.vectorize("Teszt mondat.")
    print()
