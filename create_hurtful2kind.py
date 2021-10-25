# Setup
import nltk
# Uncomment if not downloaded.
# nltk.download('wordnet')
from nltk.corpus import wordnet as wn
import pandas as pd
import random
import json
import pickle
import pdb
import json
import re
import os
from tqdm import tqdm

"""
TODO: Write usage.
"""


def read_hurtlex(pth):
    # Read Hurtlex
    hurtlex = pd.read_csv(pth, sep="\t")
    return hurtlex
    
    
def create_dict(hurtlex):
    hurtful2ant = {}
    
    def get_antonym(row):
        cat = row.category
        if cat not in hurtful2ant:
            # First time adding lemma->antonym from cat
            hurtful2ant[cat] = {}
        lemma = row.lemma
        synsets = wn.synsets(lemma)
        if len(synsets) > 0:
            syn = wn.synsets(lemma)[0]
            for j in syn.lemmas():
                if j.antonyms():
                    if lemma not in hurtful2ant[cat]:
                        hurtful2ant[cat][lemma] =  j.antonyms()[0].name()
                    break
    
    hurtlex.apply(lambda x: get_antonym(x), axis=1)
    # Print statistics:
    
    print("Length of dict (# categories): ", len(hurtful2ant))
    for i,v in hurtful2ant.items():
        print(f"Length of dict {i}: ", len(hurtful2ant[i]))
        random_keys = random.sample(list(hurtful2ant[i]), min(3, len(hurtful2ant[i])))
        print("Randomly sampled pairs: ", [(k, hurtful2ant[i][k]) for k in random_keys])
        
    return hurtful2ant


def read_train_data(pth):
    # Read GPT-2 train data and swap words 
    train_data = []
    with open(pth) as f:
        for line in f:
            train_data.append(json.loads(line))
    return train_data


def write_train_data(pth, train_data):
    with open(pth, 'w') as outfile:
        for entry in train_data:
            json.dump(entry, outfile)
            outfile.write('\n')
    
    
def swap_cat(train_data, cat_dct):

    def repl(match, anton):
        if match.group(0)[0].isupper():
            return anton[0].upper()+anton[1:]
        else:
            return anton

    for d in tqdm(train_data, total=len(train_data)):
        for lemma, anton in cat_dct.items():
            d['text'] = re.sub(fr"\b{lemma}\b",
                               lambda x: repl(x, anton),
                               d['text'])
            
    return train_data
    

def main():
    # Define params
    hurtlex_pth = "hurtlex-master/lexica/EN/1.2/hurtlex_EN.tsv"
    save_pth = "outputs/hurtlex2ant.pickle"
    
    # Create dict.
    hurtlex = read_hurtlex(hurtlex_pth)
    d = create_dict(hurtlex)
    with open(save_pth, "wb") as f:
        pickle.dump(d, f)
        print(f"Saved dict {save_pth}.")
        
    # Swap words.     
    swap_this = "gpt-2-output-dataset/data/webtext.train.jsonl"
    train_data = read_train_data(swap_this)
    
    # Combine all dicts into one
    mega_dict = {}
    for _, dct in d.items():
        mega_dict.update(dct)
    d["combined"] = mega_dict
    
    for cat, cat_dct in d.items():
        # Create directory, save the new data
        pth = f"/home/gridsan/akyurek/git/bias-lm/gpt-2-output-dataset/data_{cat}"
        if not os.path.exists(pth):
            os.makedirs(pth)
        swapped = swap_cat(train_data, cat_dct)
        write_train_data(os.path.join(pth, "webtext.train.jsonl"),
                         swapped)
        
    
if __name__ == "__main__":
    main()