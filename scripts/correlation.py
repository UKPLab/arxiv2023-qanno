import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr, kendalltau
import pingouin as pg

scores_a = np.array([1,2,3,4])
scores_b = np.array([3,4,5,5])

data = pd.DataFrame({
    "item": np.concatenate([np.arange(len(scores_a)), np.arange(len(scores_b))]),
    "score" : np.concatenate([scores_a, scores_b]),
    "judge": ["A"]*len(scores_a) + ["B"] * len(scores_b) }

)

pearson = pearsonr(scores_a, scores_b).statistic
spearman = spearmanr(scores_a, scores_b).correlation
kendall = kendalltau(scores_a, scores_b).correlation

icc = pg.intraclass_corr(data=data, targets='item', raters='judge',
                         ratings='score').round(3)

print(pd.pivot_table(data, index='judge', columns='item'))

print(pearson)
print(spearman)
print(kendall)
print(icc)