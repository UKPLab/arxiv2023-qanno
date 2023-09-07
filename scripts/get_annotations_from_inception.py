import contextlib
import os
import zipfile
from io import BytesIO

import joblib
from pycaprio import Pycaprio
from pycaprio.core.objects import Document, Project
from pycaprio.mappings import InceptionFormat
from tqdm import tqdm

from qanno.paths import (
    PATH_DATA,
    PATH_DATA_ANNOTATED_JSONCAS,
    PATH_DATA_ANNOTATED_XMI,
    PATH_ROOT,
)

HOST = "https://blinky.ukp.informatik.tu-darmstadt.de/inception-experimental"
PROJECT_NAME = "qanno-real-real-real"
USER = "jandalf"

# os.environ["REQUESTS_CA_BUNDLE"] = str(PATH_DATA / "cert.pem")


@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument"""

    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()


def build_client() -> Pycaprio:
    secrets = {}
    with open(PATH_ROOT / ".env") as f:
        for line in f:
            k, v = line.strip().split("=")
            secrets[k.strip()] = v.strip()

    return Pycaprio(HOST, (USER, secrets["inception-pw"]))


def find_project(client: Pycaprio) -> Project:
    projects = client.api.projects()

    for project in projects:
        if project.project_name == PROJECT_NAME:
            return project

    raise Exception(f"Did not find project with name [{PROJECT_NAME}]")


def get_annotations():
    PATH_DATA_ANNOTATED_XMI.mkdir(exist_ok=True, parents=True)
    PATH_DATA_ANNOTATED_JSONCAS.mkdir(exist_ok=True, parents=True)

    client = build_client()
    project = find_project(client)
    documents = client.api.documents(project)

    def _download_document_as_xmi(document: Document):
        exported_name = document.document_name.replace(".pdf", ".xmi")
        target_path = PATH_DATA_ANNOTATED_XMI / exported_name

        if target_path.is_file():
            return

        try:
            xmi = client.api.annotation(project, document, USER, InceptionFormat.XMI)

            zip_buffer = BytesIO(xmi)

            with zipfile.ZipFile(zip_buffer, "r", zipfile.ZIP_DEFLATED, False) as zip_file:
                zip_file.extract(exported_name)
        except Exception as e:
            print(f"Could not download [{exported_name}]", str(e))

    def _download_document_as_jsoncas(document: Document):
        exported_name = document.document_name.replace(".pdf", ".json")
        target_path = PATH_DATA_ANNOTATED_JSONCAS / exported_name

        if target_path.is_file():
            # return
            pass

        try:
            jsoncas = client.api.annotation(project, document, USER, "jsoncas")
            with open(target_path, "wb") as f:
                f.write(jsoncas)
        except Exception as e:
            print(f"Could not download [{exported_name}]", str(e))


    for document in tqdm(documents):
        _download_document_as_jsoncas(document)


def _main():
    get_annotations()


if __name__ == "__main__":
    _main()
