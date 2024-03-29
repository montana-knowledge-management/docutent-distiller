import os.path
import subprocess
import time
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Response, status, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Extra, ValidationError

from docutent_distiller.ml_project import MachineLearningProject

import logging

log_config = uvicorn.config.LOGGING_CONFIG
log_config["formatters"]["default"]["use_colors"] = True
log_config["formatters"]["access"]["use_colors"] = True

log_config["formatters"]["access"]["fmt"] = '%(asctime)s - %(levelprefix)s %(client_addr)s - %(status_code)s - %(message)s'#"%(asctime)s - %(levelprefix)s - %(message)s" \

log_config["formatters"]["default"]["fmt"] = "%(asctime)s - %(levelprefix)s - %(message)s" #'%(asctime)s - %(levelprefix)s %(client_addr)s - "%(request_line)s" %(status_code)s - %(message)s'#"%(asctime)s - %(levelprefix)s - %(message)s"
log_config["loggers"]["uvicorn.error"]["level"] = "ERROR"

logger = logging.getLogger("uvicorn")

class InputJsonML(BaseModel):
    """
    Class for validating the input sent to the /process endpoint for MachineLearningProject.
    """

    text: str

    # Setting for keeping the additional keys in the input json intact
    class Config:
        extra = Extra.allow


# Defining the API
app = FastAPI(title="{} API", docs_url="/apidocs", redoc_url=None)

tags_metadata = [
    {
        "name": "process",
        "description": "Run project on a single document sent for the API.",
        "externalDocs": {
            "description": "Find out more",
            "url": "http://montana.ai",
        },
    },
    {"name": "ping", "description": "Endpoint for pinging server."},
    {
        "name": "docs",
        "description": "Endpoint for OpenAPI documentation.",
        "externalDocs": {
            "description": "Find out more",
            "url": "http://montana.ai",
        },
    },
    {
        "name": "docs",
        "description": "Test page for the API. Endpoint called by the main page of the API test page.",
    },
    {"name": "root", "description": "Endpoint for the project documentation."},
]


@app.post("/process", include_in_schema=True, tags=["process"])
async def process(item: dict, response: Response):
    """
    Endpoint for performing the project.run() method on data sent for the API in JSON format.
    The endpoint performs automatic input validation via the Item class.
    """
    try:
        app.project.validate(item)
    except ValidationError as e:
        logger.warn(f"Input validation failed: {e.json()}")
        response.status_code = status.HTTP_400_BAD_REQUEST
        return {
            "status": "failed",
            "error": e.__class__.__name__,
            "detail": e.errors(),
        }
    else:
        try:
            app.project.add_single_input(item)
            app.project.run()
            return app.project.get_single_output()
        except Exception as e:
            response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
            return {
                "status": "failed",
                "error": e.__class__.__name__,
                "detail": repr(e),
            }


@app.get("/ping", include_in_schema=True, tags=["ping"])
def ping():
    """
    Pings the server to check if it is available.
    """
    result_json = {}
    result_json["call_time"] = time.ctime()
    result_json["msg"] = "The API is working."
    return result_json


class Encapsulator:
    """
    Server for running a custom project as an API.
    """

    def __init__(self, project: MachineLearningProject):
        """
        :param project: either MachineLearningProject or SimulationProject instance
        """
        self.app = app
        self.app.project = project
        self.app.title = self.app.title.format(project.app_name)
        # self.workers = self.number_of_workers()
        self.host = "127.0.0.1"
        self.port = 5000
        self.cert_file_path = None
        self.key_file_path = None

    def set_cert_file_path(self, cert_file_path):
        self.cert_file_path = cert_file_path

    def set_key_file_path(self, key_file_path):
        self.key_file_path = key_file_path

    def set_endpoint_for_docs(self, docs_path, endpoint="/", build=True):
        """
        Builds mkdocs documentation and deploys the documentation at the given endpoint.
        :param docs_path: path to the docs folder containing mkdocs.yml
        :param endpoint: endpoint to publish built docs to
        :param build: whether to build documentation or not
        :return:
        """
        if build:
            self.build_docs(docs_path)
        site_path = Path(docs_path).joinpath("site")
        self.app.mount(endpoint, StaticFiles(directory=site_path, html=True, check_dir=True), name="documentation")

    def build_docs(self, docs_path):
        """
        Build the documentation with mkdocs.
        """
        cwd = Path(os.getcwd())
        os.chdir(docs_path)
        subprocess.run("mkdocs build", shell=True, check=True)
        os.chdir(cwd)
        (docs_path / "site" / "images").resolve().mkdir(exist_ok=True)

    def set_host(self, host: str):
        """
        Set the IP address of the host.
        :param host: e.g. 127.0.0.1
        """
        self.host = str(host)

    def set_port(self, port: int):
        """
        Set the port.
        :param port: int, e.g. 5000
        """
        self.port = int(port)

    def run(self):
        """
        Running the application that is running the specified input project's run method.
        :return: None
        """
        if self.key_file_path and self.cert_file_path:
            uvicorn.run(
                self.app,
                host=self.host,
                port=self.port,
                log_level="info",
                ssl_keyfile=self.key_file_path,
                ssl_certfile=self.cert_file_path,
            )
        else:
            uvicorn.run(self.app, host=self.host, port=self.port, log_level="info")

    def __call__(self, *args, **kwargs):
        self.run()


if __name__ == "__main__":
    uvicorn.run("server:app", host="127.0.0.1", port=5000, log_config=log_config)
