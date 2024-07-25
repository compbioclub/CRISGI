import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from src.util import set_adata_obs
import os, random

from sksurv.nonparametric import kaplan_meier_estimator
from sksurv.compare import compare_survival


def assign_score_group(df, x, by='mean'):
    if by == 'median':
        cutoff = df['score'].quantile(0.5)
    else:
        cutoff = df['score'].mean()
    if x <= cutoff:
        return f'<= {by}'
    return f'> {by}'


def survival_analysis(sniee_obj,
                      relations=None,
                      survival_types = ['os', 'pfs'],
                      time_unit = 'time',
                      test_type='trend', method='pearson',
                      title='', out_prefix='test'):
    adata = sniee_obj.adata
    edata = sniee_obj.edata
    if relations is None:
        relations = edata.uns[f'{method}_{test_type}_relations']
    sedata = edata[edata.obs['type'] == 'Test', relations]
    sedata.obs['score'] = sedata.layers[f'{method}_entropy'].sum(axis=1)
    sedata.obs['score_group'] = sedata.obs['score'].apply(lambda x: assign_score_group(sedata.obs, x))
    df = sedata.obs.copy()

    for survival in survival_types:
        if survival not in df.columns:
            continue
        df = df[~df[f'{survival}_status'].isna()]
        df = df[~df[survival].isna()]
        df[f"{survival}_status"] = df[f"{survival}_status"].astype(bool)
        if df[[f"{survival}_status", 'score_group']].value_counts().shape[0] == 1:
            continue
        for score_group in df['score_group'].unique():
            mask_group = df["score_group"] == score_group
            time_treatment, survival_prob_treatment, conf_int = kaplan_meier_estimator(
                df[f"{survival}_status"][mask_group],
                df[survival][mask_group],
                conf_type="log-log",
            )
            if score_group.startswith('<'):
                color = 'steelblue'
            else:
                color = 'red'
            plt.step(time_treatment, survival_prob_treatment, where="post", label=f"score {score_group}", color=color)
            plt.fill_between(time_treatment, conf_int[0], conf_int[1], alpha=0.25, step="post", color=color)

        dt = np.dtype([(f"{survival}_status", bool), (survival, float)])
        y = [(df.iloc[i][f"{survival}_status"], df.iloc[i][survival]) for i in range(df.shape[0])]
        y = np.array(y, dtype=dt)
        chi2, p_value = compare_survival(y, df["score_group"])

        plt.ylim(0, 1)
        plt.ylabel(r"est. probability of survival $\hat{S}(t)$")
        plt.xlabel(f"{survival.upper()} {time_unit} $t$")
        plt.legend(loc="best")
        plt.title(f'{title} log-rank test\nchi2: {round(chi2, 2)}, p-value: {p_value}')
        print(out_prefix)
        if out_prefix:
            plt.savefig(f'{out_prefix}_{survival.upper()}_surv.png')
            plt.show()
        else:
            plt.show()
    
    return sedata