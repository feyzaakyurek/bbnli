# Command: python test_nli.py roberta data/nli/gender/man_is_to_programmerv2.csv

# pose sequence as a NLI premise and label as a hypothesis
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import pandas as pd
import ipdb
from torch import argmax
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import sys
import os
from tqdm import tqdm

# bart_entail_dict = {0:1, 1:2, 2:0} # In mnli, 0 means false, 1 neutral, 2 entailment.
t0_entail_dict = {"Yes":0, "No":1, "Neutral":2}

if __name__ == "__main__":
    model = sys.argv[1] # model name
    pth = sys.argv[2] # csv file
    df = pd.read_csv(pth) 

    # Check if the requested predictions are already there.
    if f"preds_{model}" in df.columns:
        print(f"Predictions for {model} are already in {pth}.")
        # exit(0) 

    premise = df['premise'].tolist()
    hypothesis = df['hypothesis'].tolist()
    device = "cuda:0"
    
    if model == "bart":
        nli_model = AutoModelForSequenceClassification.from_pretrained('bart-large-mnli').to(device)
        tokenizer = AutoTokenizer.from_pretrained('bart-large-mnli')
    elif model == "t03b":
        tokenizer = AutoTokenizer.from_pretrained("T0_3B")
        nli_model = AutoModelForSeq2SeqLM.from_pretrained("T0_3B").to(device)
    elif model == "roberta":
        tokenizer = AutoTokenizer.from_pretrained("roberta-large-mnli")
        nli_model = AutoModelForSequenceClassification.from_pretrained("roberta-large-mnli").to(device)
    else:
        raise ValueError(f"Model name {model} not known.")
    
    preds = []
    # Run through model pre-trained on MNLI
    print(f"Reading {os.path.split(pth)[1]}.")
    print("Iterating over premises ...")
    for ind, pre in enumerate(tqdm(premise)):
        if model == "bart":
            x = tokenizer.encode(pre, hypothesis[ind], return_tensors='pt',
                             truncation='only_first').to(device)
            output = argmax(nli_model(x)[0]).item()
            # output = bart_entail_dict[output]
        elif model == "t03b":
            text = f"Sentence 1: {pre}\n Sentence 2: {hypothesis[ind]}\nQuestion: Does Sentence 1 entail Sentence 2? No, Yes, or Neutral?" # an example prompt.
            x = tokenizer(text, return_tensors='pt').input_ids.to(device)
            output = nli_model.generate(input_ids=x)[0]
            output = tokenizer.decode(output, skip_special_tokens=True)
            if output in t0_entail_dict:
                output = t0_entail_dict[output]
            else:
                output = -1
        else:
            text = f"{pre} </s></s> {hypothesis[ind]}"
            x = tokenizer.encode(text, return_tensors='pt').to(device)
            output = argmax(nli_model(x)[0]).item()
            # output = bart_entail_dict[output]
        preds.append(output)
    
    df[f"preds_{model}"] = preds
    df.to_csv(pth, index=False)
    