import unittest
from fastapi.testclient import TestClient

from docutent_distiller.encapsulator import Encapsulator
from docutent_distiller.ml_project import MachineLearningProject
from importlib_resources import files


class DummyMLandSimulationProject(MachineLearningProject):
    _input = []
    _output = []

    def run(self):
        self._output_data = self._input_data
        self._output = self._input

    def update_input(self):
        pass


class TestEncapsulator(unittest.TestCase):
    # HTTP client
    example_project = DummyMLandSimulationProject(app_name="test_name")
    server = Encapsulator(example_project)
    server.set_endpoint_for_docs(files("docutent_distiller") / "docs")
    client = TestClient(server.app)

    # HTTPS client
    server2 = Encapsulator(example_project)
    server2.set_key_file_path("some_path")
    server2.set_cert_file_path("some_path")

    def test_empty_input(self):
        wrong_json = {}
        response = self.client.post("/process", json=wrong_json, headers={"Content-Type": "application/json"})
        self.assertIsNot(response.status_code, 200)
        self.assertDictEqual(
            response.json(),
            {'status': 'failed', 'error': 'ValidationError',
             'detail': [{'loc': ['text'], 'msg': 'field required', 'type': 'value_error.missing'}]},
        )

    def test_http_ping(self):
        response = self.client.get("/ping")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["msg"], "The API is working.")
        self.assertIn("call_time", response.json())

    def test_http_root(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)

    def test_server(self):
        self.assertEqual(self.server.host, "127.0.0.1")
        self.assertEqual(self.server.port, 5000)
        self.assertEqual(self.server.cert_file_path, None)
        self.assertEqual(self.server.key_file_path, None)
        self.assertEqual(self.server.app.title, "test_name API")

    def test_missing_text_key_ml(self):
        wrong_json = {"test": "Without Text key."}
        response = self.client.post("/process", json=wrong_json, headers={"Content-Type": "application/json"})
        self.assertIsNot(response.status_code, 200)
        self.assertDictEqual(
            response.json(),
            {'status': 'failed', 'error': 'ValidationError',
             'detail': [{'loc': ['text'], 'msg': 'field required', 'type': 'value_error.missing'}]},
        )

    def test_with_text_key_ml(self):
        good_json = {"text": "With text key."}
        response = self.client.post("/process", json=good_json, headers={"Content-Type": "application/json"})
        self.assertTrue(response.status_code, 200)
        self.assertDictEqual(response.json(), good_json)

    def test_https_paths(self):
        self.assertIsNotNone(self.server2.cert_file_path)
        self.assertIsNotNone(self.server2.key_file_path)

    def test_set_host(self):
        example_host = "1.2.3.4"
        # serv = Server(self.example_project)
        self.server.set_host(example_host)
        self.assertEqual(self.server.host, example_host)

    def test_set_port(self):
        example_port = 123
        self.server.set_port(example_port)
        self.assertEqual(self.server.port, example_port)
