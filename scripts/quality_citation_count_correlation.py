import json
from typing import List, Dict, Any, TypeVar, Generator

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from more_itertools import chunked
from requests import Session
from scipy.stats import kendalltau

from qanno.paths import PATH_DATA_GENERATED
import seaborn as sns

def get_paper_batch(session: Session, ids: list[str], fields: str = 'paperId,citationCount', **kwargs) -> list[dict]:
    params = {
        'fields': fields,
        **kwargs,
    }
    body = {
        'ids': ids,
    }

    # https://api.semanticscholar.org/api-docs/graph#tag/Paper-Data/operation/post_graph_get_papers
    with session.post('https://api.semanticscholar.org/graph/v1/paper/batch',
                       params=params,
                       json=body) as response:
        response.raise_for_status()
        return response.json()

def get_paper_infos(ids: list[str], batch_size: int = 100, **kwargs) -> list[dict]:
    result = []

    # use a session to reuse the same TCP connection
    with Session() as session:
        # take advantage of S2 batch paper endpoint
        for ids_batch in chunked(ids, n=batch_size):
            result.extend(get_paper_batch(session, ids_batch, **kwargs))

    return result


def load_data_relevant() -> List[Dict[str, Any]]:
    with open(PATH_DATA_GENERATED / "relevant_clean.json", "r") as f:
        return json.load(f)

def main():
    data = load_data_relevant()
    ids = [f"ACL:{x['uid']}" for x in data]

    ratings = [x["overall_score"] for x in data]
    citation_counts = [x.get('citationCount', np.nan) if x is not None else np.nan for x in get_paper_infos(ids)]

    tau = kendalltau(citation_counts, ratings,  nan_policy="omit")

    sns.histplot(x=[x for x in citation_counts if not np.isnan(x) and x > 0 ], log_scale=True)
    plt.gca().xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    plt.gca().set_xticks([1, 10, 100, 150])

    print(tau)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()