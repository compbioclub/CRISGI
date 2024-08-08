import pandas as pd
import anndata as ad
import os
import pickle
import src.plotting as pl
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed

# It is a script to find the biomarkers for flu datasets.
# The script will generate the landscape score and the relation score for each dataset.
# And generate the training dataset's results for rest validation dataset.
# The results will be saved in the ./data/flu folder.
# The folder "TER_top$NUM1" in each compressed package stands for the top $NUM1 edges with the highest scores.
# In each "TER_top$NUM1" folder, the file named "*TER$NUM2_landscape.*" represents the total $NUM2 edges with the "TER=TRUE" flag among the top $NUM1 edges.

def generate_list(start, step1, step2, midpoint, end):
    result = []
    value = start
    while value < midpoint:
        result.append(value)
        value += step1
    while value + step2 < end:
        result.append(value)
        value += step2
    result.append(end)
    return result


def load_sniee_obj(dataset, file_name, sniee_obj_dict={}, from_dict=True):
    if dataset not in sniee_obj_dict:
        sniee_obj = pickle.load(open(file_name, "rb"))
        if from_dict:
            sniee_obj_dict[dataset] = sniee_obj
    else:
        sniee_obj = sniee_obj_dict[dataset]
    return sniee_obj


def process_top_n_relations(
    n_top_relations_steps,
    sniee_obj,
    method,
    test_type,
    train_dataset,
    datasets,
    plot_label,
    rep_n,
    sniee_obj_dict,
    from_dict,
):
    relations = sniee_obj.edata.uns[f"{method}_{test_type}"][:n_top_relations_steps]
    nrow, ncol = 3, 4
    fig, axes = plt.subplots(nrow, ncol, figsize=(12, 10))
    fig.suptitle(f"Train {train_dataset}")
    axes = axes.reshape(-1)
    i = 0
    for val_dataset in datasets:
        if train_dataset == val_dataset:
            flag = "train"
        else:
            flag = "val"
        print(flag, val_dataset)
        out_dir = f"./data/flu/{train_dataset}/{flag}_{val_dataset}/{test_type}_top{n_top_relations_steps}"
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        val_sniee_obj = load_sniee_obj(
            dataset=val_dataset,
            file_name=f"./data/flu/{val_dataset}/{val_dataset}_sniee_obj.pk",
            sniee_obj_dict=sniee_obj_dict,
            from_dict=from_dict,
        )
        filtered_relations = val_sniee_obj.test_val_trend_entropy(
            relations,
            method=method,
            p_cutoff=0.05,
            out_prefix=f"{out_dir}/{flag}_{val_dataset}",
        )
        pl.get_landscape_score(val_sniee_obj, filtered_relations, out_dir=out_dir)
        pl.generate_landscape_images(
            folder_path=out_dir,
            output_path=out_dir,
            rep_n=rep_n,
            robust=True,
            scale=False,
        )
        for col in plot_label:
            pl.relation_score(
                val_sniee_obj,
                groupby=col,
                relations=filtered_relations,
                test_type=test_type,
                title=f"{flag}\n",
                ax=axes[i],
            )
            i += 1
        del val_sniee_obj
        del filtered_relations

    fig.tight_layout()
    fig.savefig(
        f"./data/flu/{train_dataset}/{train_dataset}_relation_{n_top_relations_steps}_score.png"
    )
    del relations


sniee_obj_dict = {}

datasets = [
    "GSE30550_H3N2",
    "GSE52428_H3N2",
    "GSE52428_H1N1",
    "GSE73072_H3N2_DEE2",
    "GSE73072_H3N2_DEE5",
    "GSE73072_H1N1_DEE3",
    "GSE73072_H1N1_DEE4",
    "GSE73072_HRV_DUKE",
    "GSE73072_HRV_UVA",
    "GSE73072_RSV_DEE1",
]

n_top_relations = None
test_type = "TER"
method = "pearson"
plot_label = ["symptom"]
from_dict = True
rep_n = 1
# Please allocate enough memory for the number of threads, each thread will use 40GB of memory
n_threads = 16

for i, train_dataset in enumerate(datasets):
    print("--------", train_dataset)

    out_dir = f"./data/flu/{train_dataset}"

    sniee_obj = load_sniee_obj(
        dataset=train_dataset,
        file_name=f"./data/flu/{train_dataset}/{train_dataset}_sniee_obj.pk",
        sniee_obj_dict=sniee_obj_dict,
        from_dict=from_dict,
    )

    sniee_obj.test_DER(groupby="symptom")
    sniee_obj.get_DER(
        per_group="Symptomatic",
        n_top_relations=n_top_relations,
        p_adjust=True,
        p_cutoff=0.05,
        fc_cutoff=1,
        sortby="pvals_adj",
    )
    sniee_obj.test_TER(per_group="Symptomatic")

    sniee_obj.n_threads = n_threads

    sniee_obj.pathway_enrich(n_top_relations=n_top_relations, n_space=10, method=method)
    pl.pathway_dynamic(
        sniee_obj, p_adjust=True, p_cutoff=0.05, n_top_pathway=30, method=method
    )

    if n_top_relations is None:
        n_top_relations = len(sniee_obj.edata.uns[f"{method}_{test_type}"])

    with ProcessPoolExecutor(max_workers=n_threads) as executor:
        futures = [
            executor.submit(
                process_top_n_relations,
                n_top_relations_steps,
                sniee_obj,
                method,
                test_type,
                train_dataset,
                datasets,
                plot_label,
                rep_n,
                sniee_obj_dict,
                from_dict,
            )
            for n_top_relations_steps in generate_list(
                start=10, step1=10, step2=50, midpoint=300, end=n_top_relations
            )
        ]

        for future in as_completed(futures):
            try:
                future.result()
            except Exception as exc:
                print(f"Generated an exception: {exc}")
    del sniee_obj
