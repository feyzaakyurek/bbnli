from bbnli_inference_bias import get_bias_scores
import matplotlib.pyplot as plt
from json2csv import json2csv
import seaborn as sns
import pandas as pd
import pickle
import glob
import os

overwrite_csv = False
skip_inference = False
num_gens = 5
model = "t0pp"

if __name__ == "__main__":

    for domain in ["religion", "gender", "race"]:
        # Create the outputs directory
        os.makedirs(f"outputs/bbnli/{domain}", exist_ok=True)

        # Populate csv files using json topics.
        files = glob.glob(f'data/bbnli/{domain}/*.json')
        for file in files:
            print(f"Creating csv from json: {file}")
            json2csv(file, domain, overwrite=overwrite_csv)

        # Load the big inference dict for the domain
        bigdqa, bigdnli = {}, {}
        bigdqapth = f"outputs/bbnli/{domain}/{model}_bigd-qa-inference-table.pkl"
        bigdnlipth = f"outputs/bbnli/{domain}/{model}_bigd-nli-inference-table.pkl"
        
        if os.path.exists(bigdqapth):
            with open(bigdqapth, "rb") as input_file:
                bigdqa = pickle.load(input_file)
        if os.path.exists(bigdnlipth):
            with open(bigdnlipth, "rb") as input_file:
                bigdnli = pickle.load(input_file)

        # Create predictions and compute bias scores
        if not skip_inference:
            files = glob.glob(f'outputs/bbnli/{domain}/*.csv')
            for file in files:
                print(f"Creating predictions and computing bias: {file}")
                bigdqa, bigdnli = get_bias_scores(file, model, bigdqa, bigdnli, num_gens, skip_inference=skip_inference)

        # Save bigdqa, bigdnli
        with open(bigdqapth, "wb") as f:
            pickle.dump(bigdqa, f)
        with open(bigdnlipth, "wb") as f:
            pickle.dump(bigdnli, f)

    # Visualize results
    ll = []
    keys = []
    for domain in ["race", "religion", "gender"]:
        files = glob.glob(f'outputs/bbnli/{domain}/*-{model}-n{num_gens}-bias-results.tsv')
        for file in files:
            catname = os.path.split(file)[-1].split(f"-{model}")[0]
            keys.append(catname)
            df = pd.read_csv(file, sep="\t")
            df["Domain"] = domain.capitalize()
            print(catname)
            ll.append(df)

    bigdf = pd.concat(ll, keys=keys)

    hh = bigdf.reset_index().loc[:,["level_0", "Domain", "Task", "BiasScore"]]
    hh = hh.rename(columns={"level_0":"Bias Subtopic"})
    hh = (hh.pivot(index=["Domain", "Bias Subtopic"], values="BiasScore", columns="Task") * 100).round(2)

    sns.set(rc={'figure.figsize':(2.7,10.27)})
    ax = sns.heatmap(hh.sort_values(["Domain", "QA"]), cmap="crest", annot=True)

    for i in range(19):
        if i == 6 or i == 11:
            ax.axhline(i, color='white', lw=5)
    plt.show()
    ax.get_figure().savefig(f"outputs/bbnli/new_heatmap_numgens_{num_gens}_{model}.pdf",  bbox_inches="tight") 