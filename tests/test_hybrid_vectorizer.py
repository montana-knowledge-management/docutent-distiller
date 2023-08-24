import os
import unittest
from unittest import mock

from docutent_distiller.hybrid_bert_vectorizer import HybridVectorizer
import numpy as np


class MyTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.vectorizer = HybridVectorizer()

    def test_almost_equal_vectors(self):
        text = "Ez egy teszt mondat."
        short_vec = self.vectorizer.short_vectorizer.get_cls_token_embedding([text])[0]
        long_vec = self.vectorizer.long_vectorizer.vectorize(text)
        self.assertTrue(np.allclose(short_vec, long_vec, atol=1e-5))

    def test_long_sentence(self):
        text = "Ez egy {} hosszú mondat.".format(
            " ".join(["igen nagyon fű de nagyon tartalmas és unalmas és hosszú"] * 100)
        )
        short_vec = self.vectorizer.short_vectorizer.get_cls_token_embedding([text])[0]
        long_vec = self.vectorizer.vectorize([text])[0]
        self.assertFalse(np.allclose(short_vec, long_vec, atol=1e-5))

    def test_get_vector_str(self):
        text = "Ez egy {} hosszú mondat.".format(
            " ".join(["igen nagyon fű de nagyon tartalmas és unalmas és hosszú"] * 100)
        )
        vector = self.vectorizer.get_vector(text)

        self.assertEqual(vector.shape, (768,))

    def test_get_vector_list(self):
        text = "Ez egy {} hosszú mondat.".format(
            " ".join(["igen nagyon fű de nagyon tartalmas és unalmas és hosszú"] * 100)
        )
        vector = self.vectorizer.get_vector([text, text], sentence_avg=False)

        self.assertEqual(vector.shape, (2, 768))

    def test_get_vector_list_avg(self):
        text = "Ez egy {} hosszú mondat.".format(
            " ".join(["igen nagyon fű de nagyon tartalmas és unalmas és hosszú"] * 100)
        )
        vector = self.vectorizer.get_vector([text, text], sentence_avg=True)

        self.assertEqual(vector.shape, (768,))

    def test_batching(self):
        sentences = [
            "Ez egy teszt mondat.",
            "Ez egy {} hosszú mondat.".format(
                " ".join(["igen nagyon fú de nagyon tartalmas és unalmas és hosszú"] * 100)
            ),
            "Ez egy másik harmadik teszt mondat.",
        ]

        with mock.patch.dict(os.environ, {"BATCH_SIZE": "2"}):
            self.vectorizer = HybridVectorizer()
            vectors_batch_2 = self.vectorizer.vectorize(sentences)

        with mock.patch.dict(os.environ, {"BATCH_SIZE": "1"}):
            self.vectorizer = HybridVectorizer()
            vectors_batch_1 = self.vectorizer.vectorize(sentences)

        self.assertEquals(vectors_batch_2.shape, (3, 768))
        self.assertTrue(np.allclose(vectors_batch_1, vectors_batch_2, atol=1e-5))


if __name__ == "__main__":
    unittest.main()
