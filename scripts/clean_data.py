import json
from pathlib import Path
from typing import Any, Dict, List

from qanno.paths import PATH_DATA_GENERATED


def load_data_all() -> List[Dict[str, Any]]:
    with open(PATH_DATA_GENERATED / "all.json", "r") as f:
        return json.load(f)


def fix_single(name_in: str, name_out: str):
    with open(PATH_DATA_GENERATED / name_in, "r") as f:
        data = json.load(f)

    remaps = {
        "ExpertFeedback": "GiveAnnotatorsFeedback",
        "AutomaticCheck": "AutomaticChecks",
        "AnnotatorFeedback": "AnnotatorDebriefing",
        "FixBad": "Correction",
        "GoldsetQA": "ControlQuestions",
        "IterativeRefinement": "ImproveGuidelines",
        "Loop": "AgileAnnotation",
        "QualityCheckOrDeboard": "DeboardAnnotators",
        "ExpertFiltering": "ManualFilter",
        "OrthogonalAnnotation": "IndirectValidation",
    }

    removes = {
        "AgreementForRectify",
        "AnnotatorBalance",
        "AnnotatorDiversity",
        "Batch",
        "BatchReannotate",
        "Cleanup",
        "Confidence",
        "ErrorAnalysis",
        "ExpertQA",
        "ModelInTheLoop",
        "MultiStep",
        "Preannotation",
        "Reannotation",
    }

    for paper in data:
        paper["manual_annotation"] = paper.pop("relevant")
        assert "relevant" not in paper

        if not paper["manual_annotation"]:
            continue

        qm = set(paper["quality_management"])

        for remove in removes:
            if remove in qm:
                qm.remove(remove)

        new_qm = set()
        for method in qm:
            if method in remaps:
                new_qm.add(remaps[method])
            else:
                new_qm.add(method)

        paper["quality_management"] = list(sorted(new_qm))

    with open(PATH_DATA_GENERATED / name_out, "w") as f:
        json.dump(data, f, indent=2)


def main():
    fix_single("all.json", "all_clean.json")
    fix_single("relevant.json", "relevant_clean.json")


if __name__ == "__main__":
    main()
