import unittest
from docutent_distiller.sbert_embedding import SBertEmbedding
from sklearn.metrics.pairwise import cosine_similarity


test_text = ["Ez egy példa magyar szöveg.", "Ez még egy példa szöveg.", "Ez teljesen másról szól."]


class MyTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.sbert_vectorizer = SBertEmbedding("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

    def test_init(self):
        self.assertIsNotNone(self.sbert_vectorizer.tokenizer)
        self.assertIsNotNone(self.sbert_vectorizer.model)

    def test_get_vector_string(self):
        vector = self.sbert_vectorizer.get_vector(test_text[0], sentence_avg=True)

        self.assertEqual(vector.shape, (384,))

    def test_get_vector_list_avg(self):
        vector = self.sbert_vectorizer.get_vector(test_text, sentence_avg=True)

        self.assertEqual(vector.shape, (384,))

    def test_get_vector_list(self):
        vector = self.sbert_vectorizer.get_vector(test_text, sentence_avg=False)

        self.assertEqual(vector.shape, (3, 384))


if __name__ == "__main__":
    unittest.main()
