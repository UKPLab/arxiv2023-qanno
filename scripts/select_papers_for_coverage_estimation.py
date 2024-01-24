import random
import time
from collections import defaultdict

import numpy as np
import wget
from loguru import logger
from pathvalidate import sanitize_filename
from tqdm import tqdm

from qanno.anthology import AclAnthology
from qanno.paths import PATH_DATA_PDFS_SELECTED_COVERAGE


def select_papers():
    anthology = AclAnthology()

    rng = np.random.default_rng(23)

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

    papers_by_year = defaultdict(list)
    min_year = 2013
    max_year = 2022

    print()

    for paper in anthology.papers:
        year = paper.event.year

        if year < min_year:
            continue

        if year > max_year:
            continue

        if paper.event.venue.acronym not in target_acronyms:
            continue

        papers_by_year[year].append(paper)


    for year, papers in tqdm(papers_by_year.items()):
        if year != 2013:
            continue

        selected = rng.choice(papers, size=50, replace=False)

        for paper in selected:
            file_name = f"{paper.uid}.pdf"
            file_name = sanitize_filename(file_name)

            path = PATH_DATA_PDFS_SELECTED_COVERAGE / str(year) / file_name
            path.parent.mkdir(exist_ok=True, parents=True)

            if not path.exists():
                logger.info(f"Downloading PDF for [{file_name}]")

                try:
                    wget.download(paper.pdf_url, str(path))
                    time.sleep(2)
                except Exception as e:
                    logger.error(f"Could not download [{paper.title}] [{paper.pdf_url}]")
                    continue


def main():
    select_papers()

if __name__ == '__main__':
    main()