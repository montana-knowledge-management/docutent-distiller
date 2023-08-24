import unittest
from docutent_distiller.bert_vectorizer import BertVectorizerCLS
from sklearn.metrics.pairwise import cosine_similarity


test_text = ["Ez egy példa magyar szöveg.", "Ez még egy példa szöveg.", "Ez teljesen másról szól."]


class MyTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.bert_vectorizer = BertVectorizerCLS()

    def test_init(self):
        self.assertIsNotNone(self.bert_vectorizer.tokenizer)
        self.assertIsNotNone(self.bert_vectorizer.model)

    def test_get_tokens_number(self):
        token_ids = self.bert_vectorizer.get_tokens_number(test_text[0])
        self.assertEqual(6, token_ids)
        token_ids, tokens = self.bert_vectorizer.get_tokens_number(test_text[0], return_tokens=True)
        self.assertEqual([["[CLS]", "Ez", "egy", "példa", "magyar", "szöveg", ".", "[SEP]"]], tokens)

    def test_get_cls_token_embedding(self):
        vectors = self.bert_vectorizer.get_cls_token_embedding(test_text)
        self.assertEqual(768, len(vectors[0]))
        cos_sim = cosine_similarity(vectors[:1], vectors[1:])
        self.assertGreater(cos_sim[0][0], cos_sim[0][1])

    def test_get_encoding_dict(self):
        dicts = self.bert_vectorizer.get_encoding_dict(test_text[0])
        self.assertEqual(["input_ids", "token_type_ids", "attention_mask"], list(dicts.keys()))

    def test_get_vector_string(self):
        vector = self.bert_vectorizer.get_vector(test_text[0], sentence_avg=True)

        self.assertEqual(vector.shape, (768,))

    def test_get_vector_list_avg(self):
        vector = self.bert_vectorizer.get_vector(test_text, sentence_avg=True)

        self.assertEqual(vector.shape, (768,))

    def test_get_vector_list(self):
        vector = self.bert_vectorizer.get_vector(test_text, sentence_avg=False)

        self.assertEqual(vector.shape, (3, 768))


if __name__ == "__main__":
    unittest.main()
