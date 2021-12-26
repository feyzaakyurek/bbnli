from hf_inference_api_nlibias import get_nlibias_scores
import matplotlib.pyplot as plt
from json2csv import json2csv
import seaborn as sns
import pandas as pd
import pickle
import glob
import os
import ipdb



if __name__ == "__main__":
    domain = "race"
    num_gens = 5
    model = "t0pp"

    # Populate csv files using json topics.
    files = glob.glob(f'data/nli/{domain}/*.json')
    for file in files:
        print(f"Creating csv from json: {file}")
        json2csv(file)

    # Load the big inference dict for the domain
    bigdqa, bigdnli = {}, {}
    bigdqapth = f"outputs/nlibias/{domain}/{model}_bigd-qa-inference-table.pkl"
    bigdnlipth = f"outputs/nlibias/{domain}/{model}_bigd-nli-inference-table.pkl"
    if os.path.exists(bigdqapth):
        with open(bigdqapth, "rb") as input_file:
            bigdqa = pickle.load(input_file)
    if os.path.exists(bigdnlipth):
        with open(bigdnlipth, "rb") as input_file:
            bigdnli = pickle.load(input_file)

    # Create predictions and compute bias scores
    files = glob.glob(f'outputs/nlibias/{domain}/{model}_*.csv')
    # pth = f"outputs/nlibias/{domain}/"
    # files = [pth+el for el in []]

    for file in files:
        print(f"Creating predictions and computing bias: {file}")
        bigdqa, bigdnli = get_nlibias_scores(file, model, bigdqa, bigdnli, num_gens)

    # Save bigdqa, bigdnli
    with open(bigdqapth, "wb") as f:
        pickle.dump(bigdqa, f)
    with open(bigdnlipth, "wb") as f:
        pickle.dump(bigdnli, f)

    # Visualize results
    files = glob.glob(f'outputs/nlibias/{domain}/{model}_*-n{num_gens}-bias-results.tsv')
    ll = []
    keys = []

    for file in files:
        catname = os.path.split(file)[-1].split(" - ")[0]
        keys.append(catname)
        df = pd.read_csv(file, sep="\t")
        ll.append(df)

    bigdf = pd.concat(ll, keys=keys)
    bigdf.to_csv(f"outputs/nlibias/{domain}/{model}_bias_scores.tsv", sep="\t")
    hh = bigdf.reset_index().loc[:,["level_0", "Task", "BiasScore"]]
    hh = hh.rename(columns={"level_0":"Bias Category"})
    hh = hh.pivot(index='Bias Category', columns='Task', values='BiasScore').sort_values("NLI")
    g = sns.heatmap(hh.sort_values("NLI"), cmap="crest", annot=True)
    g.get_figure().savefig(f"outputs/nlibias/{domain}/{model}_heatmap_numges_{num_gens}.jpg",  bbox_inches="tight") 
    plt.figure()