"""
Bert Long vectorizer works slowly and fails on short sentences. Hence, this class uses BertVectorizerCLS when it is
possible and uses LongVectorizer when necessary.
"""
import os
from typing import Union, List

import numpy as np
from docutent_distiller.bert_vectorizer import BertVectorizerCLS
from docutent_distiller.long_document_embedder import BertLongVectorizer


class HybridVectorizer:
    def __init__(self, model="SZTAKI-HLT/hubert-base-cc"):
        self.short_vectorizer = BertVectorizerCLS()
        self.long_vectorizer = BertLongVectorizer(model_name=model)
        self.batch_size = int(os.environ.get("BATCH_SIZE", 1))

    def vectorize(self, sentences: list):
        vectors = []
        np_sentences = np.array(sentences)
        long_mask = np.array([len(tokens) > 510 for tokens in self.short_vectorizer.tokenizer(sentences)["input_ids"]])

        short_vectors = []
        long_vectors = []

        for sents in batch(np_sentences[~long_mask], self.batch_size):
            short_vectors.extend(list(self.short_vectorizer.get_cls_token_embedding(sents.tolist())))

        for sent in np_sentences[long_mask]:
            long_vectors.append(self.long_vectorizer.vectorize(sent, matrix=False))

        for m in long_mask:
            if m:
                vectors.append(long_vectors.pop(0))
            else:
                vectors.append(short_vectors.pop(0))

        vectors = np.array(vectors)
        return vectors

    def get_vector(self, sentences: Union[List[str], str], sentence_avg: bool = True):
        if type(sentences) == str:
            sentences = [sentences]
        vectors = self.vectorize(sentences)
        if sentence_avg:
            return np.array(vectors).mean(axis=0)
        return vectors


def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx : min(ndx + n, l)]
