import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import cassis
import numpy as np
import pandas as pd
from cassis import Cas, TypeSystem
from tqdm import tqdm

from qanno.paths import (
    PATH_DATA_ANNOTATED,
    PATH_DATA_ANNOTATED_JSONCAS,
    PATH_DATA_GENERATED,
    PATH_DATA_SELECTED_PAPERS_CSV,
)


def check_paper_selection():
    df_papers = pd.read_csv(PATH_DATA_SELECTED_PAPERS_CSV)

    selected_papers = set(df_papers["uid"])
    annotated_papers = set(p.stem.split("_", maxsplit=1)[0] for p in PATH_DATA_ANNOTATED_JSONCAS.iterdir())

    for uid in annotated_papers:
        if uid not in selected_papers:
            print("Was in INCEpTION but should be not", uid)

    for uid in selected_papers:
        if uid not in annotated_papers:
            print("Should be in INCEpTION but is not", uid)


def select_single(cas: Cas, type_name: str) -> Any:
    result = cas.select(type_name)
    assert len(result) == 1
    return result[0]


def parse_jsoncas(p: Path) -> Optional[Dict[str, Any]]:
    result: Dict[str, Any] = {
        "uid": p.stem.split("_", maxsplit=1)[0],
        "name": p.stem.split("_")[1],
    }

    with p.open("rb") as f:
        ts = TypeSystem(add_document_annotation_type=True)
        cas = cassis.load_cas_from_json(f, typesystem=ts)

    general = select_single(cas, "webanno.custom.General")

    result["relevant"] = general.get("Relevant") or False
    if not result["relevant"]:
        return result

    # Task types
    task_types = set(general.get("TaskType").elements)

    result["has_annotation"] = "Annotation" in task_types
    result["has_text_production"] = "TextProduction" in task_types

    result["num_annotators"] = general.get("NumAnnotators").elements
    result["annotators"] = general.get("Annotators").elements

    if not result["has_annotation"] and result["has_text_production"]:
        result["num_annotators"] = ["N/A"]

    # Overall
    assert general.get("Overall") is not None, f"No overall judgement set: {result['name']}"
    result["overall"] = general.get("Overall")[3:]
    result["overall_score"] = 3 - int(general.get("Overall")[0]) + 1

    result["guidelines_available"] = general.get("GuidelinesAvailable")
    schema_creation = general.get("GuildelineCreation")
    if schema_creation == "Existing":
        result["schema_new"] = False
    else:
        result["schema_new"] = True

    result["tools"] = general.get("Tools.elements") or []

    # Quality Management Methods
    qm_methods = set(general.get("QualityControlMethod").elements)
    qm_methods = _post_process_quality_management_methods(qm_methods)

    # Adjudication Methods
    adjudication_methods = general.get("AggregationMethod").elements

    if len(adjudication_methods) == 0:
        if result["has_annotation"]:
            adjudication_methods.append("?")
        elif result["has_text_production"]:
            adjudication_methods.append("N/A")

    result["adjudication_methods"] = adjudication_methods

    # Validation
    result["validation"] = _parse_validation(cas)
    if result["validation"]["uses_validation"]:
        qm_methods.add("ValidateAnnotations")

    # Agreement
    result["agreement"] = _parse_agreements(cas)
    if result["agreement"]["uses_agreement"]:
        qm_methods.add("Agreement")

    # Error Rates
    result["error_rate"] = _parse_error_rates(cas)
    if result["error_rate"]["uses_error_rate"]:
        qm_methods.add("ErrorRate")

    result["quality_management"] = list(qm_methods)

    return result


def _parse_validation(cas: Cas) -> Dict[str, Any]:
    validations = cas.select("webanno.custom.Validation")

    if len(validations) == 0:
        return {"uses_validation": False}

    validation = {"uses_validation": True}
    entries = []

    for v in validations:
        a = v.get("Annotators.elements")

        if a is None or len(a) == 0:
            annotator = "?"
        else:
            assert len(a) == 1
            annotator = a[0]

        sample_size = v.get("SampleSize")
        total_size = v.get("TotalSize")
        only_subset = not (sample_size == 0 and total_size == 0)
        e = {
            "uses_validation": True,
            "sample_size": sample_size,
            "total_size": total_size,
            "only_subset": only_subset,
            "annotator": annotator,
        }

        entries.append(e)

    validation["entries"] = entries

    return validation


def _parse_agreements(cas: Cas) -> Dict[str, Any]:
    agreements = cas.select("webanno.custom.AgreementValue")

    if len(agreements) == 0:
        return {"uses_agreement": False}

    agreement = {"uses_agreement": True}
    entries = []

    for a in agreements:
        i = a.get("Interpretation.elements")

        if i is None or len(i) == 0:
            interpretations = ["None"]
        else:
            interpretations = i

        sample_size = a.get("SampleSize")
        total_size = a.get("TotalSize")
        only_subset = not (sample_size == 0 and total_size == 0)
        value = a["Value"]

        if sample_size < 0:
            sample_size = np.nan

        if total_size < 0:
            total_size = np.nan

        if value < 0:
            value = np.nan

        e = {
            "method": a.get("Method") or "?",
            "value": value,
            "interpretations": interpretations,
            "only_subset": only_subset,
        }

        if only_subset:
            e["sample_size"] = sample_size
            e["total_size"] = total_size

        entries.append(e)

    agreement["entries"] = entries

    return agreement


def _parse_error_rates(cas: Cas) -> Dict[str, Any]:
    error_rates = cas.select("webanno.custom.ErrorRateValue")

    if len(error_rates) == 0:
        return {"uses_error_rate": False}

    result = {"uses_error_rate": True}
    entries = []

    for a in error_rates:
        sample_size = a.get("SampleSize")
        total_size = a.get("TotalSize")
        only_subset = not (sample_size == 0 and total_size == 0)
        value = a["Value"]

        if sample_size < 0:
            sample_size = np.nan

        if total_size < 0:
            total_size = np.nan

        if value < 0:
            value = np.nan

        e = {
            "value": value,
            "only_subset": only_subset,
        }
        if only_subset:
            e["sample_size"] = sample_size
            e["total_size"] = total_size

        entries.append(e)

    result["entries"] = entries

    return result


def _post_process_quality_management_methods(methods: Set[str]) -> Set[str]:
    m = set(methods)

    if "SubsetAgreement" in m:
        m.remove("SubsetAgreement")
        m.add("Agreement")

    return m


def automatically_check_annotations(result: Dict[str, Any]):
    assert result["relevant"] is not None
    if not result["relevant"]:
        return

    assert result["has_annotation"] or result["has_text_production"]
    assert result["num_annotators"] is not None
    assert result["annotators"] is not None and len(result["annotators"]) > 0

    assert result["overall"] in ["Sufficient", "Underwhelming", "Very Good"]
    assert result["overall_score"] in [1, 2, 3]

    assert result["has_annotation"] or result["has_text_production"]
    if result["has_text_production"] and not result["has_annotation"]:
        assert result["num_annotators"] == ["N/A"]

    if result["has_annotation"]:
        annotators = result["annotators"]
        num_annotators = result["num_annotators"].copy()

        if "Algorithmic" in annotators or result["has_text_production"]:
            num_annotators.remove("N/A")

        if len(num_annotators) > 0:
            assert "?" in num_annotators or len([x for x in num_annotators if x.isnumeric()]) > 0

    # Adjudication Methods
    if len(result["adjudication_methods"]) > 0:
        assert result["has_annotation"] or result["adjudication_methods"] == ["N/A"]

    if not result["has_annotation"] and result["has_text_production"]:
        assert result["adjudication_methods"] == ["N/A"]

    if result["has_annotation"] and result["has_text_production"]:
        assert "N/A" not in result["adjudication_methods"]

    if result["has_annotation"] and "?" in result["adjudication_methods"]:
        assert len(result["adjudication_methods"]) == 1

    if result["has_annotation"]:
        assert "1" in result["num_annotators"] or result["adjudication_methods"] != ["N/A"] or result["annotators"] == ["Algorithmic"]

    # Check that validation is in quality_management if a validation step is explicitly defined and vice versa
    assert result["validation"]["uses_validation"] == ("ValidateAnnotations" in result["quality_management"])

    if result["validation"]["uses_validation"]:
        for v in result["validation"]["entries"]:

            assert v["uses_validation"] is True
            assert v["sample_size"] is not None
            assert v["total_size"] is not None
            assert v["only_subset"] is not None
            assert v["annotator"] is not None

    # Check that agreement is in quality_management if agreement is computed and vice versa
    assert result["agreement"]["uses_agreement"] == ("Agreement" in result["quality_management"])

    if result["agreement"]["uses_agreement"]:

        for v in result["agreement"]["entries"]:

            assert v["only_subset"] is not None
            assert v["only_subset"] == False or v["only_subset"] == True

            if v["only_subset"]:
                assert v["sample_size"] is not None
                assert v["total_size"] is not None

                assert v["sample_size"] >= 0 or np.isnan(v["sample_size"])
                assert v["total_size"] >= 0 or np.isnan(v["total_size"])
            else:
                assert "sample_size" not in v
                assert "total_size" not in v

            assert v["method"] is not None
            assert v["interpretations"] is not None
            assert len(v["interpretations"]) >= 1

            assert v["value"] is not None
            assert isinstance(v["value"], (int, float, complex)) and not isinstance(v["value"], bool)
            assert v["value"] >= 0 or np.isnan(v["value"])

    # Check that error rate is in quality_management if error rate is computed and vice versa
    assert result["error_rate"]["uses_error_rate"] == ("ErrorRate" in result["quality_management"])
    if result["error_rate"]["uses_error_rate"]:
        for v in result["error_rate"]["entries"]:

            assert v["only_subset"] is not None
            assert v["only_subset"] == False or v["only_subset"] == True

            if v["only_subset"]:
                assert v["sample_size"] is not None
                assert v["total_size"] is not None

                assert v["sample_size"] >= 0 or np.isnan(v["sample_size"])
                assert v["total_size"] >= 0 or np.isnan(v["total_size"])
            else:
                assert "sample_size" not in v
                assert "total_size" not in v

            assert v["value"] is not None
            assert isinstance(v["value"], (int, float, complex)) and not isinstance(v["value"], bool)
            assert np.isnan(v["value"]) or 0 < v["value"] < 50

    assert result["quality_management"] is not None
    assert result["adjudication_methods"] is not None and len(result["adjudication_methods"]) > 0


def _save_stuff(data: List[Dict[str, Any]]):
    with open(PATH_DATA_GENERATED / "all.json", "w") as f:
        json.dump(data, f)

    with open(PATH_DATA_GENERATED / "relevant.json", "w") as f:
        json.dump([x for x in data if x["relevant"]], f, indent=2)

    # Save agreement
    agreements = []
    for paper in data:
        if not paper["relevant"] or not paper["agreement"]["uses_agreement"]:
            continue

        for a in paper["agreement"]["entries"]:
            agreement = {"uid": paper["uid"], "name": paper["name"]}
            agreement.update(a)

            agreements.append(agreement)

    df_agreement = pd.DataFrame(agreements).convert_dtypes()
    df_agreement.to_csv(PATH_DATA_GENERATED / "agreement.tsv", sep="\t", index=False)

    # Save validation
    validations = []
    for paper in data:
        if not paper["relevant"] or not paper["validation"]["uses_validation"]:
            continue

        for v in paper["validation"]["entries"]:
            validation = {"uid": paper["uid"], "name": paper["name"]}
            validation.update(v)

            validations.append(validation)

    df_validation = pd.DataFrame(agreements).convert_dtypes()
    df_validation.to_csv(PATH_DATA_GENERATED / "validation.tsv", sep="\t", index=False)

    # Save error rate
    error_rates = []
    for paper in data:
        if not paper["relevant"] or not paper["error_rate"]["uses_error_rate"]:
            continue

        for a in paper["error_rate"]["entries"]:
            error_rate = {"uid": paper["uid"], "name": paper["name"]}
            error_rate.update(a)
            error_rates.append(error_rate)

    df_error_rates = pd.DataFrame(error_rates).convert_dtypes()
    df_error_rates.to_csv(PATH_DATA_GENERATED / "error_rates.tsv", sep="\t", index=False)


def _main():
    check_paper_selection()

    result = []

    files = list(PATH_DATA_ANNOTATED_JSONCAS.iterdir())
    files = list(reversed(files))

    skip = False

    for i, p in enumerate(tqdm(files)):
        if "201_SPIRS" in str(p):
            skip = False
            continue

        if skip:
            continue

        print(str(p))

        try:
            data = parse_jsoncas(p)

            automatically_check_annotations(data)
            result.append(data)

        except Exception as e:
            print()
            print(p)
            for it in data:
                print(it, data[it])
            print()
            raise e

    _save_stuff(result)


if __name__ == "__main__":
    _main()
