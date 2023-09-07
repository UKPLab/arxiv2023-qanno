import pytest as pytest

from qanno.anthology import AclAnthology


@pytest.fixture(scope="module")
def the_anthology() -> AclAnthology:
    return AclAnthology()


TEST_CASES = [
    (
        "N19-1423",
        "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding",
        "NAACL",
        2019,
    ),
    ("J02-3001", "Automatic Labeling of Semantic Roles", "CL", 2002),
    (
        "D19-1410",
        "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
        "EMNLP",
        2019,
    ),
    (
        "2015.mtsummit-users.6",
        "Yandex.Translate approach to the translation of Turkic languages [abstract",
        "MTSummit",
        2015,
    ),
]


@pytest.mark.parametrize("uid, expected_title, expected_acronym, expected_year", TEST_CASES)
def test_finding_papers_by_uid(uid, expected_title, expected_acronym, expected_year, the_anthology):
    paper = the_anthology.get_paper_by_uid(uid)

    assert paper.title == expected_title
    assert paper.event.venue.acronym == expected_acronym
    assert paper.event.year == expected_year


@pytest.mark.parametrize("expected_uid, title, expected_acronym, expected_year", TEST_CASES)
def test_finding_papers_by_uid(expected_uid, title, expected_acronym, expected_year, the_anthology):
    paper = the_anthology.get_paper_by_title(title)

    assert paper.uid == expected_uid
    assert paper.event.venue.acronym == expected_acronym
    assert paper.event.year == expected_year
