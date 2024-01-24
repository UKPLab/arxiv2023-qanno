import json
from copy import deepcopy

import pandas as pd
from cassis import Cas, load_cas_from_json
from tqdm import tqdm

from qanno.anthology import AclAnthology
from qanno.paths import PATH_DATA_ANNOTATED_COVERAGE_JSONCAS, PATH_DATA_ANNOTATED_COVERAGE, \
    PATH_DATA_PAPERS_WITH_CODE_DATASETS_JSON

rows = []

P_MAPPING_RAW = PATH_DATA_ANNOTATED_COVERAGE / "dataset_names_raw.csv"
P_MAPPING_ENRICHED = PATH_DATA_ANNOTATED_COVERAGE / "dataset_names_enriched.csv"
P_MAPPING_ENRICHED_EXCEL = PATH_DATA_ANNOTATED_COVERAGE / "dataset_names_enriched.xlsx"
P_MAPPING_ENRICHED_MAPPED = PATH_DATA_ANNOTATED_COVERAGE / "dataset_names_enriched_mapped.csv"
P_COVERAGE_DATASET_RAW = PATH_DATA_ANNOTATED_COVERAGE / "coverage_raw.json"
P_COVERAGE_DATASET_CLEAN = PATH_DATA_ANNOTATED_COVERAGE / "coverage_clean.json"

def select_first(cas: Cas, name: str):
    x = cas.select(name)
    assert len(x) == 1

    return x[0]

def prepare_raw_coverage_data_from_jsoncas():
    if P_MAPPING_RAW.is_file() and P_COVERAGE_DATASET_RAW.is_file():
        pass

    coverage_data = []
    rows_mapping = []

    for p in tqdm(list(PATH_DATA_ANNOTATED_COVERAGE_JSONCAS.iterdir())):
        with p.open("rb") as f:
            cas = load_cas_from_json(f)

        source = p.stem + ".pdf"

        introduces_dataset = select_first(cas, "webanno.custom.Relevant").introducesDataset

        if cas.typesystem.contains_type("webanno.custom.DatasetUsage"):
            datasets = [x.name for x in cas.select("webanno.custom.DatasetUsage") if x.name is not None and len(x.name.strip()) > 0]
        else:
            datasets = []

        coverage_entry = {
            "source": source,
            "introduces_dataset": introduces_dataset,
            "datasets": datasets
        }

        coverage_data.append(coverage_entry)

        for dataset in datasets:
            rows_mapping.append({"dataset": dataset})

    df = pd.DataFrame(rows_mapping).drop_duplicates(keep='first').sort_values("dataset")

    df.to_csv(P_MAPPING_RAW, index=False)

    coverage_data.sort(key=lambda x: x["source"])
    with P_COVERAGE_DATASET_RAW.open("w") as f:
        json.dump(coverage_data, f, indent=2)


def enrich_mapping():
    df = pd.read_csv(P_MAPPING_RAW)

    with PATH_DATA_PAPERS_WITH_CODE_DATASETS_JSON.open("rb") as f:
        pwc_datasets = json.load(f)

    name_to_pwc_datasets = {
        x["name"]: x for x in pwc_datasets
    }

    anthology = AclAnthology()

    result = []

    for _, row in df.iterrows():

        if not row["dataset"] or not isinstance(row["dataset"], str):
            continue

        dataset_name = row["dataset"].strip()
        mapped_name = ""
        pwc_name = ""
        pwc_url = ""
        homepage = ""

        if dataset_name.endswith(".pdf"):
            uid = dataset_name.strip().replace(".pdf", "")

            if paper := anthology.get_paper_by_uid(uid):
                mapped_name = paper.title

        if mapped_name in name_to_pwc_datasets:
            pwc_entry = name_to_pwc_datasets[mapped_name]
            pwc_name = pwc_entry["name"]
            pwc_url =  pwc_entry["url"]
            homepage = pwc_entry["homepage"]
        elif dataset_name in name_to_pwc_datasets:
            pwc_entry = name_to_pwc_datasets[dataset_name]
            pwc_name = pwc_entry["name"]
            pwc_url =  pwc_entry["url"]
            homepage = pwc_entry["homepage"]

        e = {
            "dataset_name": dataset_name,
            "mapped_name": mapped_name,
            "pwc_name": pwc_name,
            "pwc_url": pwc_url,
            "homepage": homepage
        }

        result.append(e)

    df: pd.DataFrame = pd.DataFrame(result).drop_duplicates("dataset_name").sort_values("dataset_name")
    # df.to_csv(P_MAPPING_ENRICHED, index=False)
    df.to_excel(P_MAPPING_ENRICHED_EXCEL, index=False)


def create_coverage_data():
    dataset_name_mapping = build_dataset_mapping()
    pwc_mapping = build_pwc_mapping()

    with P_COVERAGE_DATASET_RAW.open() as f:
        raw_coverage_data = json.load(f)

    result = []

    for e in raw_coverage_data:
        cleaned = deepcopy(e)

        dataset_names = [
            dataset_name_mapping[ds] if ds in dataset_name_mapping else ds for ds in e["datasets"]
        ]

        in_pwc = [
            ds in pwc_mapping for ds in dataset_names
        ]

        pwc_names = [
            pwc_mapping.get(ds, "") for ds in dataset_names
        ]

        cleaned["datasets"] = dataset_names
        cleaned["in_pwc"] = in_pwc
        cleaned["pwc_names"] = pwc_names
        result.append(cleaned)

    result.sort(key=lambda x: x["source"])
    with P_COVERAGE_DATASET_CLEAN.open("w") as f:
        json.dump(result, f, indent=2)






def build_pwc_mapping() -> dict[str, str]:
    df = pd.read_csv(P_MAPPING_ENRICHED_MAPPED, delimiter=";")

    pwc_mapping = {}

    for _, row in df.iterrows():
        dataset_name = row["dataset_name"].strip()

        if not isinstance(row["pwc_name"], str):
            continue

        pwc_name = row["pwc_name"].strip()

        if len(pwc_name) == 0:
            continue

        pwc_mapping[dataset_name] = pwc_name

        if isinstance(row["mapped_name"], str) and len(row["mapped_name"].strip()):
            pwc_mapping[dataset_name] = row["mapped_name"].strip()

    return pwc_mapping


def build_dataset_mapping() -> dict[str, str]:
    df = pd.read_csv(P_MAPPING_ENRICHED_MAPPED, delimiter=";")

    mapping = {}

    for _, row in df.iterrows():
        if not isinstance(row["mapped_name"], str):
            continue

        dataset_name = row["dataset_name"].strip()
        mapped_name = row["mapped_name"].strip()

        if len(mapped_name) > 0:
            assert dataset_name not in mapping or mapping[dataset_name] == mapped_name
            assert mapped_name not in mapping, (dataset_name, mapped_name)

            mapping[dataset_name] = mapped_name

    return mapping


if __name__ == '__main__':
    prepare_raw_coverage_data_from_jsoncas()
    enrich_mapping()
    create_coverage_data()