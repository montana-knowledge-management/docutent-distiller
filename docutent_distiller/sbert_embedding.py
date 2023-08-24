from typing import List, Union

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer


class SBertEmbedding:
    def __init__(self, model_name: str):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device=self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model.max_seq_length = 126
        self.size_mapper = {
            "sentence-transformers/distiluse-base-multilingual-cased-v2": 512,
            "sentence-transformers/paraphrase-multilingual-mpnet-base-v2": 768,
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2": 384,
        }
        self.tokens = None
        self.connected_sw_tokens = None
        self.slicing_points = None
        self.slices = None

    def get_vector(self, sentences: Union[List[str], str], sentence_avg: bool = True):
        if type(sentences) == str:
            sentences = [sentences]
        sentence_vectors = []
        for sentence in sentences:
            if sentence == "":
                sentence_vectors.append(np.zeros(self.size_mapper[self.model_name]))
                continue

            self.tokens = self.tokenizer.encode_plus(sentence, add_special_tokens=False)
            word_ends = self._search_word_ends()
            if len(word_ends) == 0:
                sentence_vectors.append(self.model.encode(sentence))
            else:
                self.slicing_points = self._search_slicing_points(word_ends)

                previous = 0
                slices = []
                for position in self.slicing_points:
                    slices.append(self.tokens["input_ids"][previous : position + 1])
                    previous = position + 1

                self.slices = slices
                vectors = []
                for slice in self.slices:
                    # print(self.tokenizer.decode(slice))
                    # print("\n\n__________\n\n")
                    text_part = self.tokenizer.decode(slice)
                    vector = self.model.encode(text_part)
                    vectors.append(vector)
                    # print(vector)
                result = np.stack(vectors, axis=0)
                result = result.mean(axis=0)
                sentence_vectors.append(result)
                # print(result.shape)
        if sentence_avg:
            result = np.array(sentence_vectors).mean(axis=0)
            return result
        return np.array(sentence_vectors)

    def _search_word_ends(self) -> List[int]:
        self.connected_sw_tokens = self.tokens.word_ids()
        word_ends = []

        for index, sw in enumerate(self.connected_sw_tokens):
            if index == 0:
                continue
            if not self.connected_sw_tokens[index] == self.connected_sw_tokens[index - 1]:
                word_ends.append(index - 1)
            if index == len(self.connected_sw_tokens) - 1:
                word_ends.append(index)

        return word_ends

    def _search_slicing_points(self, word_end_indices: List[int]) -> List[int]:
        slice_end_positions = []
        backtrack_from = self.model.max_seq_length
        backtrack_to = 0

        if word_end_indices[-1] >= self.model.max_seq_length:
            while backtrack_from <= word_end_indices[-1]:
                for possible_slice in reversed(range(backtrack_to, backtrack_from)):
                    if possible_slice in word_end_indices:
                        # if (backtrack_from + self.max_segment_length) <= word_end_indices[-1]:
                        slice_end_positions.append(possible_slice)
                        backtrack_to = backtrack_from + 1
                        backtrack_from = possible_slice + self.model.max_seq_length
                        break

        slice_end_positions.append(word_end_indices[-1])

        return slice_end_positions
