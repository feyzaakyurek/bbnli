from hf_inference_api_nlibias import get_nlibias_scores
import matplotlib.pyplot as plt
from json2csv import json2csv
import seaborn as sns
import pandas as pd
import glob
import os

if __name__ == "__main__":
    files = glob.glob('data/nli/gender/*.json')
    for file in files:
        print(f"Creating csv from json: {file}")
        json2csv(file)

    files = glob.glob('data/nli/gender/*.csv')
    for file in files:
        print(f"Creating predictions and computing bias: {file}")
        get_nlibias_scores(file)

    files = glob.glob('data/nli/gender/*bias-results.tsv')
    ll = []
    keys = []
    for file in files:
        catname = os.path.split(file)[-1].split(" - ")[0]
        keys.append(catname)
        df = pd.read_csv(file, sep="\t")
        ll.append(df)

    bigdf = pd.concat(ll, keys=keys)
    bigdf.to_csv("outputs/nlibias/bias_scores.csv")
    hh = bigdf.reset_index().loc[:,["level_0", "Task", "BiasScore"]]
    hh = hh.rename(columns={"level_0":"Bias Category"})
    hh = hh.pivot(index='Bias Category', columns='Task', values='BiasScore').sort_values("NLI")
    g = sns.heatmap(hh.sort_values("NLI"), cmap="crest", annot=True)
    g.get_figure().savefig("outputs/nlibias/heatmap.pdf") 
    plt.figure()