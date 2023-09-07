import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
from urllib.parse import urlparse

import yaml
from tqdm import tqdm

from qanno.paths import PATH_DATA_ACL_ANTHOLOGY_XML, PATH_DATA_ACL_ANTHOLOGY_YAML


@dataclass(frozen=True)
class Venue:
    name: str
    acronym: str
    oldstyle_letter: Optional[str] = None


@dataclass(frozen=True)
class Event:
    venue: Venue
    year: int


@dataclass(frozen=True)
class Paper:
    title: str
    abstract: Optional[str]
    uid: str
    pdf_url: str
    event: Event


class AclAnthology:
    def __init__(self):
        venues = self._parse_venues()
        papers = self._collect_papers(venues)

        uid_to_paper = {paper.uid: paper for paper in papers}
        title_to_paper = {_normalize_string(paper.title): paper for paper in papers}

        # assert len(uid_to_paper) == len(title_to_paper), f"{len(uid_to_paper)} != {len(title_to_paper)}"
        self.venues = venues
        self.papers = papers
        self._uid_to_paper = uid_to_paper
        self._title_to_paper = title_to_paper

    def get_paper_by_uid(self, uid: str) -> Optional[Paper]:
        return self._uid_to_paper.get(uid)

    def get_paper_by_title(self, paper_name: str) -> Optional[Paper]:
        return self._title_to_paper.get(_normalize_string(paper_name))

    def _parse_venues(self) -> List[Venue]:
        result = []

        for venue_file in (PATH_DATA_ACL_ANTHOLOGY_YAML / "venues").iterdir():
            with venue_file.open() as f:
                e = yaml.safe_load(f)

                name = e["name"]
                acronym = e["acronym"]
                oldstyle_letter = e.get("oldstyle_letter")

                venue = Venue(name=name, acronym=acronym, oldstyle_letter=oldstyle_letter)
                result.append(venue)

        return result

    def _collect_papers(self, venues: List[Venue]):
        all_papers = []
        paths = list(PATH_DATA_ACL_ANTHOLOGY_XML.iterdir())
        pbar = tqdm(paths)

        for p in pbar:
            file_name = p.name
            pbar.set_postfix_str(file_name)

            for venue in venues:
                needle = f".{venue.acronym.lower()}.xml"

                matches = False

                if needle in file_name:
                    matches = True
                elif venue.oldstyle_letter and re.match(f"{venue.oldstyle_letter}\\d+\.xml", file_name):
                    matches = True

                if matches:
                    # Parse year
                    year = _infer_year(p.stem)
                    event = Event(venue=venue, year=int(year))

                    papers = self._load_papers_for_venue(event, p)
                    all_papers.extend(papers)

        return all_papers

    def _load_papers_for_venue(self, event: Event, path_to_xml: Path) -> List[Paper]:
        tree = ET.parse(path_to_xml)
        collection = tree.getroot()

        collection_id = collection.attrib["id"]

        results = []

        for volume in collection.findall("volume"):
            volume_id = volume.attrib["id"]

            for paper in volume.findall("paper"):
                title_node = paper.find("title")
                title = "".join(title_node.itertext())

                # uid = f"{collection_id}-{paper_id}"

                paper_id = paper.attrib.get("id")
                uid = build_anthology_id(collection_id, volume_id, paper_id)

                if (url_node := paper.find("url")) is not None:
                    raw_url = url_node.text
                    url = _infer_url(raw_url)
                    if not url.endswith(".pdf"):
                        url += ".pdf"
                else:
                    url = f"https://aclanthology.org/{uid}.pdf"

                abstract_node = paper.find("abstract")

                if abstract_node is not None:
                    abstract = "".join(abstract_node.itertext())
                else:
                    abstract = None

                assert title is not None and len(title) > 0, uid

                entry = Paper(title=title, abstract=abstract, uid=uid, pdf_url=url, event=event)
                results.append(entry)

        return results


def _is_newstyle_id(anthology_id: str) -> int:
    # Taken from https://github.com/acl-org/acl-anthology/blob/02e8987747ad88504e3c20c6a6fbe16dc127976f/bin/anthology/utils.py#L37
    return anthology_id[0].isdigit()  # New-style IDs are year-first


def _infer_year(collection_id: str) -> str:
    """Infer the year from the collection ID.
    Many paper entries do not explicitly contain their year.  This function assumes
    that the paper's collection identifier follows the format 'xyy', where x is
    some letter and yy are the last two digits of the year of publication.

    Taken from https://github.com/acl-org/acl-anthology/blob/02e8987747ad88504e3c20c6a6fbe16dc127976f/bin/anthology/utils.py#L293
    """
    if _is_newstyle_id(collection_id):
        return collection_id.split(".")[0]

    assert len(collection_id) == 3, f"Couldn't infer year: unknown volume ID format '{collection_id}' ({type(collection_id)})"
    digits = collection_id[1:]
    if int(digits) >= 60:
        year = f"19{digits}"
    else:
        year = f"20{digits}"

    return year


def build_anthology_id(collection_id: str, volume_id: str, paper_id: Optional[str] = None) -> str:
    """
    Transforms collection id, volume id, and paper id to a width-padded
    Anthology ID. e.g., ('P18', '1', '1') -> P18-1001.

    Taken from
    https://github.com/acl-org/acl-anthology/blob/02e8987747ad88504e3c20c6a6fbe16dc127976f/bin/anthology/utils.py
    """
    if _is_newstyle_id(collection_id):
        if paper_id is not None:
            return f"{collection_id}-{volume_id}.{paper_id}"
        else:
            return f"{collection_id}-{volume_id}"
    # pre-2020 IDs
    if collection_id[0] == "W" or collection_id == "C69" or (collection_id == "D19" and int(volume_id) >= 5):
        anthology_id = f"{collection_id}-{int(volume_id):02d}"
        if paper_id is not None:
            anthology_id += f"{int(paper_id):02d}"
    else:
        anthology_id = f"{collection_id}-{int(volume_id):01d}"
        if paper_id is not None:
            anthology_id += f"{int(paper_id):03d}"

    return anthology_id


def _infer_url(filename: str):
    """If URL is relative, return the full Anthology URL.
    Returns the canonical URL by default, unless a different
    template is provided.

    Taken from
    https://github.com/acl-org/acl-anthology/blob/master/bin/anthology/utils.py#L268
    """

    template = "https://aclanthology.org/{}"

    if urlparse(filename).netloc:
        return filename
    return template.format(filename)


def _normalize_string(s: Optional[str]) -> str:
    if not s:
        return ""

    x = re.sub(r"[^a-zA-Z0-9 ]", "", s)
    x = " ".join(i.strip() for i in x.split())
    x = x.lower()

    return x
