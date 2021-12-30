from hf_inference_api_nlibias import get_bias_scores
import matplotlib.pyplot as plt
from json2csv import json2csv
import seaborn as sns
import pandas as pd
import pickle
import glob
import os
import ipdb

overwrite_csv = False
skip_inference = True

if __name__ == "__main__":

    num_gens = 5
    for domain in ["religion"]:
        for model in ["t0"]:

            # Populate csv files using json topics.
            files = glob.glob(f'data/nli/{domain}/*.json')
            for file in files:
                print(f"Creating csv from json: {file}")
                json2csv(file, overwrite=overwrite_csv)

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
            if not skip_inference:
                files = glob.glob(f'data/nli/{domain}/*.csv')
                for file in files:
                    print(f"Creating predictions and computing bias: {file}")
                    bigdqa, bigdnli = get_bias_scores(file, model, bigdqa, bigdnli, num_gens, skip_inference=skip_inference)

            # Save bigdqa, bigdnli
            with open(bigdqapth, "wb") as f:
                pickle.dump(bigdqa, f)
            with open(bigdnlipth, "wb") as f:
                pickle.dump(bigdnli, f)

            # Visualize results
            files = glob.glob(f'data/nli/{domain}/*-{model}-n{num_gens}-bias-results.tsv')
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
            hh = hh.pivot(index='Bias Category', columns='Task', values='BiasScore').sort_values("QA")
            g = sns.heatmap(hh.sort_values("NLI"), cmap="crest", annot=True)
            g.get_figure().savefig(f"outputs/nlibias/{domain}/{model}_heatmap_numges_{num_gens}.jpg",  bbox_inches="tight") 
            plt.figure()