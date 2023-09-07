import gzip
import hashlib
import json
import re
import shutil
import time
from dataclasses import dataclass
from typing import List
from urllib.error import HTTPError

import pandas as pd
import wget
from loguru import logger
from pathvalidate import sanitize_filename
from tqdm import tqdm

from qanno.anthology import AclAnthology
from qanno.paths import *


@dataclass
class DatasetPaper:
    uid: str
    dataset_name: str
    paper_title: str
    pdf_url: str
    venue: str
    year: int


def select_dataset_papers_from_papers_with_code() -> List[DatasetPaper]:
    files = [
        (
            PATH_DATA_PAPERS_WITH_CODE_DATASETS_GZ,
            PATH_DATA_PAPERS_WITH_CODE_DATASETS_JSON,
            "57193271ad26d827da3666e54e3c59dc",
        ),
        (
            PATH_DATA_PAPERS_WITH_CODE_PAPERS_GZ,
            PATH_DATA_PAPERS_WITH_CODE_PAPERS_JSON,
            "4531a8b4bfbe449d2a9b87cc6a4869b5",
        ),
        (
            PATH_DATA_PAPERS_WITH_CODE_LINKS_GZ,
            PATH_DATA_PAPERS_WITH_CODE_LINKS_JSON,
            "424f1b2530184d3336cc497db2f965b2",
        ),
    ]

    for source, target, md5hash in files:
        h = hashlib.new("md5")

        with source.open("rb") as f:
            h.update(f.read())

        actual_md5_hash = h.hexdigest()
        assert md5hash == actual_md5_hash, f"PWC - actual != expected: {actual_md5_hash} != {md5hash}"

        with gzip.open(source, "rb") as f_in:
            with target.open("wb") as f_out:
                shutil.copyfileobj(f_in, f_out)

    with PATH_DATA_PAPERS_WITH_CODE_DATASETS_JSON.open("rb") as f:
        datasets = json.load(f)

    with PATH_DATA_PAPERS_WITH_CODE_PAPERS_JSON.open("rb") as f:
        papers = json.load(f)
        paper_url_to_paper = {paper["paper_url"]: paper for paper in papers}

    pwc_datasets = [dataset for dataset in datasets if "Texts" in dataset["modalities"] and len(dataset["modalities"]) == 1]

    # Deduplicate based on lowe case name, there are some duplicates
    # that cause problems later if we do not filter them
    dataset_names = set()
    deduplicated_datasets = []
    for dataset in sorted(pwc_datasets, key=lambda x: x["name"]):
        dataset_name = dataset["name"].lower()
        if dataset_name in dataset_names:
            continue

        deduplicated_datasets.append(dataset)
        dataset_names.add(dataset_name)

    pwc_datasets = deduplicated_datasets

    target_acronyms = {
        "ACL",
        "CL",
        "EMNLP",
        "NAACL",
        "TACL",
        "EACL",
        "COLING",
        "LREC",
        "CoNLL",
        "AACL",
        "findings",
    }
    target_acronyms = {e.lower() for e in target_acronyms}

    selected_papers = []
    missing_information = 0
    filtered_out = 0
    not_found = 0
    not_found_venues = set()
    filtered_out_venues = set()

    anthology = AclAnthology()

    for dataset in pwc_datasets:
        if dataset.get("paper") is None:
            missing_information += 1
            continue

        if dataset["paper"].get("url") is None:
            missing_information += 1
            continue

        url = dataset["paper"]["url"]

        if url not in paper_url_to_paper:
            missing_information += 1
            continue

        paper = paper_url_to_paper[url]
        event_name = paper["proceeding"]

        year = int(paper["date"].split("-")[0])

        title = paper["title"]
        paper_from_anthology = anthology.get_paper_by_title(title)

        if paper_from_anthology is None:
            logger.warning(f"Did not find [{title}] in ACL Anthology")
            not_found += 1

            if event_name is not None:
                not_found_venues.add(event_name)
            continue

        acronym = paper_from_anthology.event.venue.acronym.lower()
        if acronym not in target_acronyms:
            # logger.warning(f"Filtered out [{paper_from_anthology.title}] from [{acronym}] at the last minute.")
            filtered_out += 1
            filtered_out_venues.add(acronym)

            continue

        assert abs(year - paper_from_anthology.event.year) <= 2, f"{year} != {paper_from_anthology.event.year} - {title}"

        dataset_paper = DatasetPaper(
            uid=paper_from_anthology.uid,
            dataset_name=dataset["name"],
            paper_title=title,
            pdf_url=paper_from_anthology.pdf_url,
            venue=paper_from_anthology.event.venue.acronym,
            year=year,
        )

        selected_papers.append(dataset_paper)

    print("Total papers:", len(datasets))
    print("Texts papers:", len(pwc_datasets))
    print("Target conferences:", len(selected_papers))
    print("Missing Information:", missing_information)
    print("Filtered out:", filtered_out)
    print("Not Found", not_found)

    for v in sorted(filtered_out_venues):
        # print(v)
        pass

    for v in sorted(not_found_venues):
        # print(v)
        pass

    PATH_DATA_GENERATED.mkdir(exist_ok=True, parents=True)
    df = pd.DataFrame(selected_papers)
    df.to_csv(PATH_DATA_SELECTED_PAPERS_CSV, index=False)
    df.to_excel(PATH_DATA_SELECTED_PAPERS_XLSX)

    # df = pd.DataFrame(selected_papers)
    # df.to_excel(PATH_ROOT / "papers.xlsx", index=False)

    return selected_papers


def crawl_dataset_papers(papers: List[DatasetPaper]):
    PATH_DATA_PDFS.mkdir(exist_ok=True, parents=True)
    PATH_DATA_PDFS_SELECTED.mkdir(exist_ok=True, parents=True)

    logger.info("Crawling papers!")

    # Clean up

    for f in PATH_DATA_PDFS_SELECTED.iterdir():
        if f.is_file():
            f.unlink()

    for paper in tqdm(papers):
        dataset_name = paper.dataset_name.replace(" ", "_")
        file_name = f"{paper.uid}_{dataset_name}.pdf"
        file_name = sanitize_filename(file_name)

        path = PATH_DATA_PDFS / file_name

        if not path.exists():
            # logger.debug(f"PDF for [{file_name}] already exists, skipping...")

            logger.info(f"Downloading PDF for [{file_name}]")

            try:
                wget.download(paper.pdf_url, str(path))
                time.sleep(5)
            except HTTPError:
                raise Exception(f"Could not download [{paper.paper_title}] [{paper.pdf_url}]")

        dest_path = PATH_DATA_PDFS_SELECTED / path.name
        if not dest_path.exists():
            shutil.copy(path, dest_path)


def _main():
    selected_papers = select_dataset_papers_from_papers_with_code()
    crawl_dataset_papers(selected_papers)


if __name__ == "__main__":
    _main()
