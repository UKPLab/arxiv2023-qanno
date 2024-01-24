

import json
from collections import defaultdict
from typing import Optional, TextIO

import pandas as pd
from matplotlib import pyplot as plt

from qanno.anthology import AclAnthology
from qanno.paths import PATH_DATA_PAPERS_WITH_CODE_PAPERS_JSON, PATH_DATA_PAPERS_WITH_CODE_EVALUATION_TABLES_JSON, \
    PATH_DATA_ANNOTATED_COVERAGE, PATH_DATA_RESULTS, PATH_DATA_GENERATED
from scripts.analyze_data import MyData

import seaborn as sns

P_COVERAGE_DATASET_CLEAN = PATH_DATA_ANNOTATED_COVERAGE / "coverage_clean.json"


def load_data_all() -> MyData:
    with open(PATH_DATA_GENERATED / "all_clean.json", "r") as f:
        return json.load(f)


def load_data_relevant() -> MyData:
    with open(PATH_DATA_GENERATED / "relevant_clean.json", "r") as f:
        return json.load(f)

def analyze_and_create_metrics_tex():
    # pct relevant
    def _write_statistic_command(out: TextIO, command: str, value: int, denominator: int, percentage: bool = True, fmt: str = "%d"):
        assert not (percentage is False and denominator is not None)

        assert value is not None and value > 0, f"Value is 0 for {command}"
        # assert percentage and  denominator is not None and denominator > 0, f"Denominator is 0 for {command}"

        tmplt = r"{%%\xspace{}}".replace("%%", fmt)
        out.write((r"\newcommand{\jckcnum%s}" + tmplt) % (command, value))
        out.write("\n")
        if percentage:
            out.write((r"\newcommand{\jckcpct%s}" + tmplt) % (command, value / denominator * 100))
            out.write("\n")

    data_quality_all = load_data_all()
    data_quality_relevant = load_data_relevant()

    names_quality_all = {x["name"] for x in data_quality_all}
    names_quality_relevant = {x["name"] for x in data_quality_relevant}

    with P_COVERAGE_DATASET_CLEAN.open() as f:
        coverage_data = json.load(f)

    df_papers_all = pd.DataFrame([
        {
            "source": x["source"],
            "introduces_dataset": x["introduces_dataset"],
            "relevant": len(x["datasets"]) > 0,
        }
        for x in coverage_data])

    df_datasets = pd.DataFrame([
        {
            "dataset_name": x["datasets"][i],
            "has_pwc": len(x["pwc_names"][i]) > 0,
            "included_in_quality_data_all": x["pwc_names"][i] in names_quality_all,
            "included_in_quality_data_relevant": x["pwc_names"][i] in names_quality_relevant,
        }
        for x in coverage_data for i in range(len(x["datasets"]))])

    df_papers = df_papers_all[df_papers_all["relevant"]]

    df_datasets_unique = df_datasets.drop_duplicates(subset="dataset_name")

    num_total = len(df_papers_all)
    num_relevant = len(df_papers)

    with (PATH_DATA_RESULTS / "coverage.tex").open("w") as f:
        _write_statistic_command(f, "relevant", num_relevant, num_total)
        _write_statistic_command(f, "introducesds", df_papers["introduces_dataset"].sum(), num_relevant)

        _write_statistic_command(f, "ds", len(df_datasets["dataset_name"]), None, percentage=False)
        _write_statistic_command(f, "uniqueds", df_datasets["dataset_name"].nunique(), None, percentage=False)

        _write_statistic_command(f, "haspwc", df_datasets_unique["has_pwc"].sum(), len(df_datasets_unique))

        _write_statistic_command(f, "inall", df_datasets_unique["included_in_quality_data_all"].sum(), len(df_datasets_unique))
        _write_statistic_command(f, "inrelevant", df_datasets_unique["included_in_quality_data_relevant"].sum(), len(df_datasets_unique))

    sns.countplot(df_datasets["dataset_name"].value_counts().reset_index(), x="dataset_name")
    # sns.countplot(df_datasets[df_datasets["has_pwc"]]["dataset_name"].value_counts().reset_index(), x="dataset_name")

    plt.show()


def collect_evaluations():
    # Collect evaluations
    with PATH_DATA_PAPERS_WITH_CODE_EVALUATION_TABLES_JSON.open("rb") as f:
        evaluations = json.load(f)

    titles = []

    counts = defaultdict(int)

    for e in evaluations:
        for ds in e.get("datasets"):

            counts[ds["dataset"]] += len(ds["sota"]["rows"])

            for row in ds["sota"]["rows"]:
                titles.append(row["paper_title"].lower().strip())

    x = set(titles)
    sns.histplot(x=list(counts.values()))
    plt.show()

    return x


def get_papers(year_cutoff: Optional[int] = None):
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

    # Collect Anthology papers
    anthology = AclAnthology()

    anthology_papers = []
    for paper in anthology.papers:
        if paper.event.venue.acronym not in target_acronyms:
            continue

        if year_cutoff and paper.event.year < year_cutoff:
            continue

        anthology_papers.append(paper)

    title_to_anthology_paper = {
        p.title.lower().strip() : p for p in anthology_papers
    }

    # Collect PwC papers
    with PATH_DATA_PAPERS_WITH_CODE_PAPERS_JSON.open("rb") as f:
        pwc_papers = json.load(f)

    matched_papers_count = 0
    evaluated_papers_count = 0
    for paper in pwc_papers:
        if paper["title"] is None:
            continue

        title = paper["title"].lower().strip()

        if title in title_to_anthology_paper:
            matched_papers_count += 1

    # Evaluations
    evaluated_papers = collect_evaluations()
    for title in evaluated_papers:

        if title in title_to_anthology_paper:
            evaluated_papers_count += 1

    if year_cutoff:
        print(f"Cutoff: {year_cutoff}")
    else:
        print("No cutoff")
    print("Matched", matched_papers_count)
    print("Evaluated", evaluated_papers_count)
    print("Total", len(anthology_papers))
    print("% matched", matched_papers_count / len(anthology_papers) * 100)
    print("% evaluated", evaluated_papers_count / len(anthology_papers) * 100)
    print()

def main():
    # analyze_and_create_metrics_tex()

    # collect_evaluations()
    get_papers(2018)
    get_papers()


if __name__ == '__main__':
    main()