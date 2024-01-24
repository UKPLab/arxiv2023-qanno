import contextlib
import os
import zipfile
from io import BytesIO
from pathlib import Path

import joblib
from pycaprio import Pycaprio
from pycaprio.core.objects import Document, Project
from pycaprio.mappings import InceptionFormat
from tqdm import tqdm

from qanno.paths import (
    PATH_DATA,
    PATH_DATA_ANNOTATED_QUALITY_JSONCAS,
    PATH_DATA_ANNOTATED_QUALITY_XMI,
    PATH_ROOT, PATH_DATA_ANNOTATED_COVERAGE_JSONCAS,
)

BLINKY_EXPERIMENTAL = "https://blinky.ukp.informatik.tu-darmstadt.de/inception-experimental"
BLINKY_STABLE = "https://blinky.ukp.informatik.tu-darmstadt.de/inception-stable"

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


def build_client(user: str, host: str) -> Pycaprio:
    secrets = {}
    with open(PATH_ROOT / ".env") as f:
        for line in f:
            k, v = line.strip().split("=")
            secrets[k.strip()] = v.strip()

    return Pycaprio(host, (user, secrets["inception-pw"]))


def find_project(client: Pycaprio, project_name: str) -> Project:
    projects = client.api.projects()

    for project in projects:
        if project.project_name == project_name:
            return project

    raise Exception(f"Did not find project with name [{project_name}]")


def get_annotations(admin_user, annotator: str, host: str, project_name: str, folder: Path):
    folder.mkdir(exist_ok=True, parents=True)

    client = build_client(admin_user, host)
    project = find_project(client, project_name)
    documents = client.api.documents(project)

    def _download_document_as_jsoncas(document: Document):
        exported_name = document.document_name.replace(".pdf", ".json")
        target_path = folder / exported_name

        if target_path.is_file():
            # return
            pass

        try:
            jsoncas = client.api.annotation(project, document, annotator, "jsoncas")
            with open(target_path, "wb") as f:
                f.write(jsoncas)
        except Exception as e:
            print(f"Could not download [{exported_name}]", str(e))


    for document in tqdm(documents):
        _download_document_as_jsoncas(document)


def _main():
    # get_annotations("jandalf", "qanno-real-real-real")
    get_annotations("klie_admin", "klie",  BLINKY_EXPERIMENTAL, "qanno-coverage", PATH_DATA_ANNOTATED_COVERAGE_JSONCAS)


if __name__ == "__main__":
    _main()
