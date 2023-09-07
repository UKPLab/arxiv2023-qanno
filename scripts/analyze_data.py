import json
from collections import Counter, defaultdict
from math import ceil
from typing import Any, Dict, List, TextIO

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator
from scipy.stats import binomtest
from tabulate import tabulate

from qanno.paths import PATH_DATA_GENERATED, PATH_DATA_RESULTS, PATH_PLOTS, PATH_DATA_SELECTED_PAPERS_CSV

MyData = List[Dict[str, Any]]

ANNOTATORS = {"Automatic", "Contractor", "Crowd", "Expert", "Unknown", "Volunteers"}
ANNOTATOR_PALETTE = {a: sns.color_palette()[i] for i, a in enumerate(sorted(ANNOTATORS))}


def load_data_all() -> MyData:
    with open(PATH_DATA_GENERATED / "all_clean.json", "r") as f:
        return json.load(f)


def load_data_relevant() -> MyData:
    with open(PATH_DATA_GENERATED / "relevant_clean.json", "r") as f:
        return json.load(f)


def general_data_to_df(data: MyData) -> pd.DataFrame:
    entries = []

    for paper in reversed(data):
        # print(paper.keys())
        annotators = paper["annotators"]
        validators = [x["annotator"] for x in paper["validation"]["entries"]] if paper["validation"]["uses_validation"] else []

        adjudication_methods = paper["adjudication_methods"]
        algorithmic = len(annotators) == 1 and "Algorithmic" in annotators

        # Agreement
        uses_agreement = paper["agreement"]["uses_agreement"]
        agreement_methods = set()
        agreement_interpretations = set()

        if uses_agreement:
            for e in paper["agreement"]["entries"]:
                agreement_methods.add(e["method"])
                agreement_interpretations.update(e["interpretations"])

        entry = {
            "uid": paper["uid"],
            "name": paper["name"],
            "id": paper["uid"],
            "relevant": paper["manual_annotation"],
            # "tasks": paper["task_type"],
            "has_annotation": paper["has_annotation"],
            "has_text_production": paper["has_text_production"],
            "quality_management": paper["quality_management"],
            "annotators": tuple(annotators),
            "validators": validators,
            "overall": paper["overall"],
            "adjudication_methods": adjudication_methods,
            "guidelines_available": paper["guidelines_available"],
            "algo_annotation_human_validation": algorithmic,
            "human_annotation": not algorithmic,
            "uses_error_rate": paper["error_rate"]["uses_error_rate"],
            "uses_error_rate_only_subset": paper["error_rate"]["uses_error_rate"],
            "uses_agreement": paper["agreement"]["uses_agreement"],
            "agreement_methods": list(agreement_methods),
            "agreement_interpretations": list(agreement_interpretations),
        }
        entries.append(entry)

    df = pd.DataFrame(entries)
    df = df.set_index(["uid", "name"])

    # print(df.head())

    return df


def load_agreement_df() -> pd.DataFrame:
    df = pd.read_csv(PATH_DATA_GENERATED / "agreement.tsv", sep="\t")

    return df


def load_error_rate_df() -> pd.DataFrame:
    df = pd.read_csv(PATH_DATA_GENERATED / "error_rates.tsv", sep="\t")

    return df


# Dataset Statistics


def compute_dataset_statistics(data_all: MyData, data_relevant: MyData):
    def _write_statistic_command(out: TextIO, command: str, value: int, denominator: int = None, percentage: bool = True, fmt: str = "%d"):
        assert not (percentage is False and denominator is not None)

        if denominator is None:
            denominator = num_relevant

        assert value is not None and value > 0, f"Value is 0 for {command}"
        assert denominator is not None and denominator > 0, f"Denominator is 0 for {command}"

        tmplt = r"{%%\xspace{}}".replace("%%", fmt)
        out.write((r"\newcommand{\jcknum%s}" + tmplt) % (command, value))
        out.write("\n")
        if percentage:
            out.write((r"\newcommand{\jckpct%s}" + tmplt) % (command, value / denominator * 100))
            out.write("\n")

    num_papers = len(data_all)
    num_no_human = num_papers - len(data_relevant)

    df_raw = general_data_to_df(data_relevant)
    num_algo_annotation_human_validation = df_raw["algo_annotation_human_validation"].sum()

    df = df_raw[df_raw["human_annotation"]].copy()
    num_relevant = len(df)

    assert num_algo_annotation_human_validation + num_no_human + num_relevant == num_papers

    df["task_annotation"] = np.logical_and(df["has_annotation"], ~df["has_text_production"])
    df["task_text_production"] = np.logical_and(~df["has_annotation"], df["has_text_production"])
    df["task_annotation_and_text_production"] = np.logical_and(df["has_annotation"], df["has_text_production"])
    df["has_loop"] = df["quality_management"].apply(lambda qm: "AgileAnnotation" in qm)
    df["has_indirect_validation"] = df["quality_management"].apply(lambda qm: "IndirectValidation" in qm)
    df["has_validation"] = df["quality_management"].apply(lambda qm: "ValidateAnnotations" in qm)
    df["has_validation"] = np.logical_or(df["has_validation"], df["has_indirect_validation"])
    df["has_iterative_refinement"] = np.logical_and(df["has_loop"], df["quality_management"].apply(lambda qm: "ImproveGuidelines" in qm))
    df["has_iterative_fix_bad"] = np.logical_and(df["has_loop"], df["quality_management"].apply(lambda qm: "Correction" in qm))
    df["has_annotator_feedback"] = df["quality_management"].apply(lambda qm: "AnnotatorDebriefing" in qm)
    df["has_pilot_study"] = df["quality_management"].apply(lambda qm: "PilotStudy" in qm)
    df["has_annotator_training"] = df["quality_management"].apply(lambda qm: "AnnotatorTraining" in qm)
    df["has_qualification_filter"] = df["quality_management"].apply(lambda qm: "QualificationFilter" in qm)
    df["has_qualification_test"] = df["quality_management"].apply(lambda qm: "QualificationTest" in qm)
    df["has_automatic_checks"] = df["quality_management"].apply(lambda qm: "AutomaticChecks" in qm)
    df["has_monetary_incentive"] = df["quality_management"].apply(lambda qm: "MonetaryIncentive" in qm)

    df["no_expert"] = df["annotators"].apply(lambda a: not ((len(a) == 1 or (len(a) == 2 and "Algorithmic" in a)) and "Expert" in a))
    num_no_expert = df["no_expert"].sum()
    num_annotraining_no_expert = np.logical_and(df["has_annotator_training"], df["no_expert"]).sum()

    # Rectifying measures
    df["has_fix_bad"] = df["quality_management"].apply(lambda qm: "Correction" in qm)
    df["has_deboard"] = df["quality_management"].apply(lambda qm: "DeboardAnnotators" in qm)
    df["has_agreement_filter"] = df["quality_management"].apply(lambda qm: "AgreementFilter" in qm)
    df["has_manual_filter"] = df["quality_management"].apply(lambda qm: "ManualFilter" in qm)
    df["has_time_filter"] = df["quality_management"].apply(lambda qm: "TimeFilter" in qm)
    df["has_data_filter"] = np.logical_or.reduce([df["has_agreement_filter"], df["has_manual_filter"], df["has_time_filter"]])
    df["has_expert_feedback"] = df["quality_management"].apply(lambda qm: "GiveAnnotatorsFeedback" in qm)
    df["has_rectifying_measure"] = np.logical_or.reduce((df["has_fix_bad"], df["has_deboard"], df["has_agreement_filter"], df["has_expert_feedback"]))

    num_given_feedback_no_expert = np.logical_and(df["has_expert_feedback"], df["no_expert"]).sum()

    # Quality Estimation
    df["has_control_questions"] = df["quality_management"].apply(lambda qm: "ControlQuestions" in qm)
    df["has_quality_estimation"] = np.logical_or.reduce((df["uses_error_rate"], df["uses_agreement"], df["has_control_questions"]))

    # Adjudication
    dfadj = df[np.logical_and(df["has_annotation"], df["adjudication_methods"].apply(lambda adj: ["N/A"] != adj))]
    # dfadj = df[df["has_annotation"]]

    df["has_majority_voting"] = dfadj["adjudication_methods"].apply(lambda adj: "MajorityVoting" in adj or "ExpertBreaksTies" in adj or "TotalAgreement" in adj)
    df["has_manual_curation"] = dfadj["adjudication_methods"].apply(lambda adj: "ExpertCuration" in adj or "ManualCuration" in adj)
    df["has_someone_break_ties"] = dfadj["adjudication_methods"].apply(lambda adj: "ExpertBreaksTies" in adj)
    df["does_not_mention_adjudication_method"] = dfadj["adjudication_methods"].apply(lambda adj: len(adj) == 0 or (len(adj) == 1 and "?" in adj))
    df["has_probabilistic_aggregation"] = dfadj["adjudication_methods"].apply(lambda adj: "DawidSkeene" in adj)

    num_adj_specified = len(dfadj[dfadj["adjudication_methods"].apply(lambda adj: ["?"] != adj)])
    num_adj_not_specified = df["does_not_mention_adjudication_method"].sum()

    assert num_adj_specified + num_adj_not_specified == len(dfadj)

    # Agreement
    df_agreement = load_agreement_df()
    num_could_use_agreement = num_relevant
    num_uses_agreement = df["uses_agreement"].sum()
    agreements_num_reported = len(df_agreement)
    agreements_only_subset = df_agreement["only_subset"].sum()
    agreement_samplesize_mean = df_agreement["sample_size"].mean()
    agreement_samplesize_median = df_agreement["sample_size"].median()
    agreement_samplesize_eqbelow100 = df_agreement[df_agreement["sample_size"] <= 100]["sample_size"].size
    agreement_samplesize_eqbelow200 = df_agreement[df_agreement["sample_size"] <= 200]["sample_size"].size

    dfagre = df[df["uses_agreement"]]["agreement_methods"].apply(lambda agm: len(agm))
    agreement_measures_mean = dfagre.mean()
    agreement_measures_median = dfagre.median()

    df["only_uses_percent_agreement"] = df["agreement_methods"].apply(lambda agm: set(agm) == {"Percentage"})
    df["has_unknown_agreement_method"] = df["agreement_methods"].apply(lambda agm: "?" in agm)
    df["uses_correlation_for_agreement"] = df["agreement_methods"].apply(lambda agm: any(x in agm for x in {"Pearson", "Spearman"}))
    df["compares_agreement_to_previous"] = df["agreement_interpretations"].apply(lambda aip: "CompareToPrevious" in aip)
    df["compares_agreement_to_literature"] = df["agreement_interpretations"].apply(lambda aip: any(x not in {"CompareToPrevious", "None", "Custom"} for x in aip))
    df["interprets_agreement_custom"] = df["agreement_interpretations"].apply(lambda aip: "Custom" in aip)
    df["no_interpretation"] = df["agreement_interpretations"].apply(lambda aip: set(aip) == {"None"})

    # Error rate
    df_error_rate = load_error_rate_df()
    error_rate_mean = df_error_rate["value"].mean()
    error_rate_median = df_error_rate["value"].median()
    error_rate_samplesize_mean = df_error_rate["sample_size"].mean()
    error_rate_samplesize_median = df_error_rate["sample_size"].median()

    error_rate_num_reported = len(df_error_rate)
    error_rate_only_subset = df_error_rate["only_subset"].sum()

    error_rates_aloghuvalid = []
    for _, row in df_raw[df_raw["algo_annotation_human_validation"]].iterrows():
        for _, e in df_error_rate[df_error_rate["uid"] == row["id"]].iterrows():
            error_rates_aloghuvalid.append(e["value"])

    error_rates_aloghuvalid_count = sum(~np.isnan(error_rates_aloghuvalid))
    error_rates_aloghuvalid_min = np.nanmin(error_rates_aloghuvalid).item()
    error_rates_aloghuvalid_max = np.nanmax(error_rates_aloghuvalid).item()
    error_rates_aloghuvalid_mean = np.nanmean(error_rates_aloghuvalid).item()
    error_rates_aloghuvalid_median = np.nanmedian(error_rates_aloghuvalid).item()

    remap = {"has_iterative_refinement": "IterativeGuidelineSchemaRefinement", "has_iterative_fix_bad": "IterativeCorrection", "annotatorfeedback": "AnnotatorDebriefing"}

    with (PATH_DATA_RESULTS / "stats.tex").open("w") as f:
        _write_statistic_command(f, "pubs", num_papers, percentage=False)
        _write_statistic_command(f, "pubnoshumanannotation", num_no_human, percentage=False)
        _write_statistic_command(f, "pubalgo", num_no_human + num_algo_annotation_human_validation, percentage=False)
        _write_statistic_command(f, "pubalgohuvalid", num_algo_annotation_human_validation, percentage=False)
        _write_statistic_command(f, "pubhashumans", len(data_relevant), percentage=False)
        _write_statistic_command(f, "pubhumanannotation", num_relevant, percentage=False)
        _write_statistic_command(f, "pubsanno", df["task_annotation"].sum())
        _write_statistic_command(f, "pubstp", df["task_text_production"].sum())
        _write_statistic_command(f, "pubstpanno", df["task_annotation_and_text_production"].sum())
        _write_statistic_command(f, "feedbackloop", df["has_loop"].sum())
        _write_statistic_command(f, "hasvalidation", df["has_validation"].sum())
        _write_statistic_command(f, "hasindirectvalidation", df["has_indirect_validation"].sum())
        _write_statistic_command(f, "iterativerefinement", df["has_iterative_refinement"].sum())
        _write_statistic_command(f, "iterativefixbad", df["has_iterative_fix_bad"].sum())
        _write_statistic_command(f, "annotatorfeedback", df["has_annotator_feedback"].sum())
        _write_statistic_command(f, "pilotstudy", df["has_pilot_study"].sum())
        _write_statistic_command(f, "guidelinesavailable", df["guidelines_available"].sum())
        _write_statistic_command(f, "annotatortraining", df["has_annotator_training"].sum())
        _write_statistic_command(f, "annotatortrainingnoexpert", num_annotraining_no_expert, denominator=num_no_expert)

        _write_statistic_command(f, "qualificationfilter", df["has_qualification_filter"].sum())
        _write_statistic_command(f, "qualificationtest", df["has_qualification_test"].sum())
        _write_statistic_command(f, "monetaryincentive", df["has_monetary_incentive"].sum())

        # Rectifying measures
        _write_statistic_command(f, "rectifyingmeasures", df["has_rectifying_measure"].sum())
        _write_statistic_command(f, "fixbad", df["has_fix_bad"].sum())
        _write_statistic_command(f, "deboard", df["has_deboard"].sum())
        _write_statistic_command(f, "agreementfilter", df["has_agreement_filter"].sum())
        _write_statistic_command(f, "manualfilter", df["has_manual_filter"].sum())
        _write_statistic_command(f, "timefilter", df["has_time_filter"].sum())
        _write_statistic_command(f, "datafilter", df["has_data_filter"].sum())
        _write_statistic_command(f, "expertfeedback", df["has_expert_feedback"].sum())
        _write_statistic_command(f, "expertfeedbacknoexpert", num_given_feedback_no_expert, denominator=num_no_expert)
        _write_statistic_command(f, "automaticchecks", df["has_automatic_checks"].sum())

        # Quality Estimation
        _write_statistic_command(f, "qualityestimation", df["has_quality_estimation"].sum())
        _write_statistic_command(f, "controlquestions", df["has_control_questions"].sum())

        # Adjudication
        _write_statistic_command(f, "adjmajorityvoting", df["has_majority_voting"].sum(), denominator=len(dfadj))
        _write_statistic_command(f, "adjbreakties", df["has_someone_break_ties"].sum(), denominator=len(dfadj))
        _write_statistic_command(f, "adjmanualcuration", df["has_manual_curation"].sum(), denominator=len(dfadj))
        _write_statistic_command(f, "adjquestionmark", df["does_not_mention_adjudication_method"].sum(), denominator=len(dfadj))
        _write_statistic_command(f, "adjprobabilistic", df["has_probabilistic_aggregation"].sum(), denominator=len(dfadj))
        _write_statistic_command(
            f,
            "adjother",
            len(dfadj) - df["has_majority_voting"].sum() - df["has_manual_curation"].sum() - df["does_not_mention_adjudication_method"].sum() - df["has_probabilistic_aggregation"].sum(),
            denominator=len(dfadj),
        )

        # Agreement
        _write_statistic_command(f, "agreement", num_uses_agreement, denominator=num_could_use_agreement)
        _write_statistic_command(f, "agreementlabeling", df[df["has_annotation"]]["uses_agreement"].sum(), denominator=num_could_use_agreement)
        _write_statistic_command(f, "agreementtp", df[df["has_text_production"]]["uses_agreement"].sum(), denominator=df["has_text_production"].sum())
        _write_statistic_command(f, "agreementnohumanannotation", df_raw[~df_raw["human_annotation"]]["uses_agreement"].sum(), percentage=False)
        _write_statistic_command(f, "agreementpercentonly", df["only_uses_percent_agreement"].sum(), denominator=num_uses_agreement)
        _write_statistic_command(f, "agreementcorrelation", df["uses_correlation_for_agreement"].sum(), denominator=num_could_use_agreement)
        _write_statistic_command(f, "agreementunknownmethod", df["has_unknown_agreement_method"].sum(), percentage=False)
        _write_statistic_command(f, "agreementinterpprevious", df["compares_agreement_to_previous"].sum(), denominator=num_uses_agreement)
        _write_statistic_command(f, "agreementcomparetoliterature", df["compares_agreement_to_literature"].sum(), denominator=num_uses_agreement)
        _write_statistic_command(f, "agreementinterpcustom", df["interprets_agreement_custom"].sum(), denominator=num_uses_agreement)
        _write_statistic_command(f, "agreementinterpnone", df["no_interpretation"].sum(), denominator=num_uses_agreement)
        _write_statistic_command(f, "agreementnumreported", agreements_num_reported, percentage=False)
        _write_statistic_command(f, "agreementonlysubset", agreements_only_subset, percentage=False)
        _write_statistic_command(f, "agreementfull", agreements_num_reported - agreements_only_subset, percentage=False)
        _write_statistic_command(f, "agreementsamplesizemean", agreement_samplesize_mean, percentage=False)
        _write_statistic_command(f, "agreementsamplesizemedian", agreement_samplesize_median, percentage=False)
        _write_statistic_command(f, "agreementsamplesizeeqbelowonehundred", agreement_samplesize_eqbelow100, denominator=agreements_only_subset)
        _write_statistic_command(f, "agreementsamplesizeeqbelowtwohundred", agreement_samplesize_eqbelow200, denominator=agreements_only_subset)
        _write_statistic_command(f, "agreementmeasurescountmean", agreement_measures_mean, percentage=False, fmt="%.2f")
        _write_statistic_command(f, "agreementmeasurescountmedian", agreement_measures_median, percentage=False)

        # Error rate
        _write_statistic_command(f, "errorrate", df["uses_error_rate"].sum())
        _write_statistic_command(f, "errorratemean", error_rate_mean, fmt="%.2f", percentage=False)
        _write_statistic_command(f, "errorratemedian", error_rate_median, fmt="%.2f", percentage=False)
        _write_statistic_command(f, "errorratesamplesizemean", error_rate_samplesize_mean, fmt="%.2f", percentage=False)
        _write_statistic_command(f, "errorratesamplesizemedian", error_rate_samplesize_median, fmt="%.2f", percentage=False)
        _write_statistic_command(f, "errorratenumreported", error_rate_num_reported, percentage=False)
        _write_statistic_command(f, "errorrateonlysubset", error_rate_only_subset, percentage=False)
        _write_statistic_command(f, "errorratesaloghuvalidcount", error_rates_aloghuvalid_count, percentage=False)
        _write_statistic_command(f, "errorratesaloghuvalidmin", error_rates_aloghuvalid_min, fmt="%.2f", percentage=False)
        _write_statistic_command(f, "errorratesaloghuvalidmax", error_rates_aloghuvalid_max, fmt="%.2f", percentage=False)
        _write_statistic_command(f, "errorratesaloghuvalidmean", error_rates_aloghuvalid_mean, fmt="%.2f", percentage=False)
        _write_statistic_command(f, "errorratesaloghuvalidmedian", error_rates_aloghuvalid_median, fmt="%.2f", percentage=False)


# Overall plot


def plot_overall_judgement_barchart(data_relevant: MyData):
    df = general_data_to_df(data_relevant)

    # Abbreviate
    # df[df["method"] == "Krippendorf"] = r"$\alpha$"

    counts = df["overall"].value_counts(normalize=True)

    df_agg = df["overall"].value_counts(normalize=True).to_frame("count").rename_axis("overall").reset_index().copy()

    df_agg["sortkey"] = 1

    df_agg.loc[df_agg["overall"] == "Underwhelming", "sortkey"] = 0
    df_agg.loc[df_agg["overall"] == "Sufficient", "sortkey"] = 1
    df_agg.loc[df_agg["overall"] == "Very Good", "sortkey"] = 2

    df_agg.sort_values(by="sortkey", inplace=True)

    df_agg.loc[df_agg["overall"] == "Underwhelming", "overall"] = "Subpar"
    df_agg.loc[df_agg["overall"] == "Sufficient", "overall"] = "Good"
    df_agg.loc[df_agg["overall"] == "Very Good", "overall"] = "Excellent"

    df_agg["count"] *= 100

    print(df_agg)

    figsize = (5, 2.5)
    plt.figure(figsize=figsize)
    g = sns.barplot(df_agg, x="overall", y="count")
    g.tick_params(bottom=False)
    g.set(xlabel=None, ylabel="%")
    # plt.xticks(rotation=5)

    plt.tight_layout()

    plt.savefig(PATH_DATA_RESULTS / "overall_barchart.pdf")

    # plt.show()


# Agreement


def plot_agreement_measure_stripplot():
    # df_distinct = df.drop_duplicates(subset=["uid", "method"])
    df = load_agreement_df()

    INTERESTING_AGREEMENTS = ["Cohen", "Fleiss", "Krippendorf", "Percentage"]
    df = df[df["method"].isin(INTERESTING_AGREEMENTS)]
    df = df[df["value"] > 0]
    df = df[df["value"] <= 1.0]

    figsize = (5, 2.5)
    plt.figure(figsize=figsize)
    ax = sns.stripplot(
        data=df,
        x="value",
        y="method",
        hue="method",
        dodge=False,
        jitter=0.25,
        marker=".",
        size=7,
        legend=False,
        order=INTERESTING_AGREEMENTS,
    )

    ax.set(xlabel=None)

    # distance across the "X" or "Y" stipplot column to span, in this case 40%
    limit_width = 0.9

    limits = {
        "Krippendorf": [0.667, 0.8],
        "Cohen": [0.2, 0.4, 0.6, 0.8],
        "Fleiss": [0.75],
    }

    for tick, text in zip(ax.get_yticks(), ax.get_yticklabels()):
        method_name = text.get_text()

        if method_name not in limits:
            continue

        for v in limits[method_name]:
            ax.plot(
                [v, v],
                [tick - limit_width / 2, tick + limit_width / 2],
                lw=1.5,
                color="black",
                linestyle="solid",
            )
    plt.tight_layout()

    plt.savefig(PATH_DATA_RESULTS / "agreement_values_stripplot.pdf")

    # plt.show()


def plot_agreement_methods_barchart():
    df = load_agreement_df()

    interesting_methods = {"Cohen", "Fleiss", "Krippendorf", "Percentage", "Classification", "?", "Correlation"}
    correlation = {"Spearman", "Pearson"}
    classification = {"F1", "Precision", "Recall", "F0.5"}

    df[df["method"] == "KrippendorfU"] = "Krippendorf"
    df["method"][df["method"].isin(correlation)] = "Correlation"
    df["method"][df["method"].isin(classification)] = "Classification"
    df = df.drop_duplicates(subset=["uid", "method"], keep="first")

    df["method"][~df["method"].isin(interesting_methods)] = "Other"

    # Abbreviate
    df[df["method"] == "Krippendorf"] = r"$\alpha$"
    df[df["method"] == "Cohen"] = r"$\kappa_c$"
    df[df["method"] == "Fleiss"] = r"$\kappa_f$"
    df[df["method"] == "Percentage"] = r"%"
    df[df["method"] == "Classification"] = r"F1/P/R"
    df[df["method"] == "Correlation"] = r"$Corr.$"

    figsize = (5, 2.5)
    plt.figure(figsize=figsize)
    g = sns.countplot(df, x="method", order=df["method"].value_counts().index)
    g.tick_params(bottom=False)
    g.set(xlabel=None, ylabel="# used")
    # plt.xticks(rotation=5)

    plt.tight_layout()

    plt.savefig(PATH_DATA_RESULTS / "agreement_barchart.pdf")

    # plt.show()


# Error Rates


def plot_error_rate_sample_sizes():
    def get_interval_width(row):
        if np.isnan(row["value"]):
            return np.nan

        if np.isnan(row["sample_size"]):
            return np.nan

        if np.isnan(row["total_size"]):
            return np.nan

        rate = row["value"]
        sample_size = int(row["sample_size"])
        k_s_obs = int(ceil(sample_size * rate / 100))
        n_pop = int(row["total_size"])

        # h = HyperCI(n_pop=n_pop, n_draw=sample_size, k_s_obs=k_s_obs)
        # res = h.ci_sim()
        # print(f"n_pop={n_pop}, n_draw={sample_size}, k_s_obs={k_s_obs}, rate={rate}")

        result = binomtest(k=k_s_obs, n=sample_size, p=rate / 100.0)

        ci = result.proportion_ci()

        return (ci.high - ci.low) / 2.0 * 100

    df = load_error_rate_df()

    # print(get_interval_width(dfs.iloc[6]))

    df.dropna(subset=["value", "sample_size"], inplace=True)
    df["half_width"] = df.apply(get_interval_width, axis=1)
    df = df[df["sample_size"] <= 1_000]

    # print(tabulate(dfs, headers="keys", showindex=False))

    figsize = (5, 2.5)
    plt.figure(figsize=figsize)
    sns.scatterplot(df, x="sample_size", y="half_width", s=15)
    plt.xlabel("# Inspected Instances")
    plt.ylabel("CI Half-Width in p.p.")

    ax = plt.gca()
    ax.xaxis.set_minor_locator(MultipleLocator(100))

    plt.tight_layout()

    plt.savefig(PATH_DATA_RESULTS / "error_rate_interval_widths.pdf")

    # plt.show()


# Annotator Statistics


def plot_annotator_statistics(data_relevant: MyData):
    df = general_data_to_df(data_relevant)

    def map_name(s):
        if s == "?":
            return "Unknown"
        elif s == "Algorithmic":
            return "Automatic"
        else:
            return s

    annotators = [map_name(y) for x in df["annotators"] for y in x]
    annotators.sort()

    figsize = (4, 2.5)
    plt.figure(figsize=figsize)
    sns.countplot(x=annotators, palette=ANNOTATOR_PALETTE)
    # plt.xlabel("Annotators")
    plt.ylabel("#")

    plt.xticks(rotation=20)

    plt.tight_layout()

    plt.savefig(PATH_DATA_RESULTS / "annotator_statistics.pdf")


def plot_validator_statistics(data_relevant: MyData):
    df = general_data_to_df(data_relevant)

    def map_name(s):
        if s == "?":
            return "Unknown"
        elif s == "Algorithmic":
            return "Automatic"
        else:
            return s

    annotators = [map_name(y) for x in df["validators"] for y in x]
    annotators.sort()

    figsize = (4, 2.5)
    plt.figure(figsize=figsize)
    sns.countplot(x=annotators, palette=ANNOTATOR_PALETTE)
    # plt.xlabel("Validators")
    plt.ylabel("#")

    plt.xticks(rotation=20)

    plt.tight_layout()

    plt.savefig(PATH_DATA_RESULTS / "validation_statistics.pdf")


def plot_distribution_over_venues():
    PATH_PLOTS.mkdir(exist_ok=True, parents=True)

    df = pd.read_csv(PATH_DATA_SELECTED_PAPERS_CSV).sort_values("venue")

    figsize = (4, 2.5)
    plt.figure(figsize=figsize)
    sns.countplot(data=df, x="venue")
    plt.xticks(rotation=45)
    plt.tight_layout()

    plt.savefig(PATH_PLOTS / "venues_countplot.pdf")

    plt.figure(figsize=figsize)
    sns.countplot(data=df, x="year")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(PATH_PLOTS / "years_countplot.pdf")

    plt.show()

def _main():
    np.random.seed(11243)

    custom_palette = sns.color_palette("Paired", 9)
    sns.palplot(custom_palette)
    sns.set_style("ticks")
    sns.set_context("notebook")
    plt.rc("axes.spines", top=False, right=False)

    PATH_DATA_RESULTS.mkdir(exist_ok=True, parents=True)

    data_all = load_data_all()
    data_relevant = load_data_relevant()
    assert len(data_all) == 591, len(data_all)

    compute_dataset_statistics(data_all, data_relevant)
    plot_overall_judgement_barchart(data_relevant)
    plot_agreement_methods_barchart()
    plot_agreement_measure_stripplot()
    plot_error_rate_sample_sizes()

    plot_annotator_statistics(data_relevant)
    plot_validator_statistics(data_relevant)

    # analyse_error_rate_sizes()

    # print_adjudication_statistics(data)
    # print_them_statistics(df_general_relevant)

    # df = data_to_agreement_df(data)
    # plot_agreement_values(df)
    # print_agreement_values_table(df)
    # calculate_agreement_statistics(data)
    # analyse_sample_sizes(df)
    # analyze_percentage_agreement_sad(df)


if __name__ == "__main__":
    _main()
