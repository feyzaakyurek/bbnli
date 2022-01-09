import json
import pandas as pd
from jinja2.nativetypes import NativeEnvironment
import os
import sys
import itertools
from collections import OrderedDict
from tqdm import tqdm
import ipdb

def json2csv(filename, domain, overwrite=False):
    # Exit if file exists.
    _, tail = os.path.split(filename)
    save_pth = os.path.join(f"outputs/nlibias/{domain}",
                            tail.replace("json", "csv"))
    if os.path.exists(save_pth):
        print(f"{save_pth} already exits.")
        if not overwrite:
            return
        else:
            print("Overwriting.")

    # Read the json file.
    print(f"Reading {filename}.")
    with open(filename, "r") as f:
        inp = json.load(f)
    env = NativeEnvironment()
    
    # Populate a csv with all possible (premise, hypothesis) combinations.
    group1 = inp["GROUP1"][0]
    group2 = inp["GROUP2"][0]
    data = OrderedDict(inp['data'])
    name = inp["name"].replace(" ", "_")
    df = pd.DataFrame(columns=["domain",
                               "name",
                               "id",
                               "type",
                               "unique_id",
                               "premise",
                               "hypothesis_type",
                               "hypothesis",
                               "question",
                               "true_label",
                               "bias_label",
                               "reference"
                               ])
    
    # Pairwise cross product data
    variables = list(itertools.product(*list(data.values())))
    keys = list(data.keys())
    
    print("Iterating over premises...")
    for ind,pre in enumerate(inp['premise']):
        for datum in variables:
            m = dict(zip(keys, datum))
            m["GROUP1"] = group1
            m["GROUP2"] = group2
            
            # Replace variables in premise.
            pre_pro = env.from_string(pre).render(**m)
            
            for ti, (test_hyp,test_hyp_l) in enumerate(inp['test_hypothesis']):

                # Replace variables in hypothesis.
                test_hyp = env.from_string(test_hyp).render(**m)
                # Replace variables in question.
                qq, _ = inp['test_question'][ti]
                qq = env.from_string(qq).render(**m)

                df.loc[len(df)] = [inp['domain'],
                                   name,
                                   f"{ind:02}",
                                   "pro",
                                   name+"_pro_"+f"{ind:02}",
                                   pre_pro,
                                   "test",
                                   test_hyp,
                                   qq,
                                   test_hyp_l,
                                   -1,
                                   inp['reference'][ind]]
                
            for ti,(hyp, label, bias_label) in enumerate(inp['bias_hypothesis_stereotypical']):
                # Replace variables in hypothesis.
                hyp = env.from_string(hyp).render(**m)
                # Replace variables in question.
                qq, *_ = inp['bias_question_stereotypical'][ti]
                qq = env.from_string(qq).render(**m)

                df.loc[len(df)] = [inp['domain'],
                                   name,
                                   f"{ind:02}",
                                   "pro",
                                   name+"_pro_"+f"{ind:02}",
                                   pre_pro,
                                   "stereotypical",
                                   hyp,
                                   qq,
                                   label,
                                   bias_label,
                                   inp['reference'][ind]]
                
            # Now anti-stereotypical formulation where group1 and group2 are swapped
            # in both premises and hypotheses.
            m["GROUP1"] = group2
            m["GROUP2"] = group1
            
            pre_anti = env.from_string(pre).render(**m)

            for ti, (test_hyp,test_hyp_l) in enumerate(inp['test_hypothesis']):
                # Replace variables in hypothesis.
                test_hyp = env.from_string(test_hyp).render(**m)
                # Replace variables in question.
                qq, *_ = inp['test_question'][ti]
                qq = env.from_string(qq).render(**m)

                df.loc[len(df)] = [inp['domain'],
                                   name,
                                   f"{ind:02}",
                                   "anti",
                                   name+"_anti_"+f"{ind:02}",
                                   pre_anti,
                                   "test",
                                   test_hyp,
                                   qq,
                                   test_hyp_l,
                                   -1,
                                   inp['reference'][ind]]
                
            for ti, (hyp, label, bias_label) in enumerate(inp['bias_hypothesis_stereotypical']):
                # Replace variables in hypothesis.
                hyp = env.from_string(hyp).render(**m)
                # Replace variables in question.
                qq, *_ = inp['bias_question_stereotypical'][ti]
                qq = env.from_string(qq).render(**m)
                df.loc[len(df)] = [inp['domain'],
                                   name,
                                   f"{ind:02}",
                                   "anti",
                                   name+"_anti_"+f"{ind:02}",
                                   pre_anti,
                                   "anti-stereotypical",
                                   hyp,
                                   qq,
                                   label,
                                   bias_label,
                                   inp['reference'][ind]]
    df = df.drop_duplicates()
    df = df.reset_index()
    df.columns = ["Index"] + df.columns[1:].tolist()
    print(f"Length of the resulting full csv: {len(df)}.")
    df.to_csv(save_pth, index=False)

if __name__ == "__main__":
    filename = sys.argv[1] # e.g. "data/nli/gender/man_is_to_programmer.json"
    domain = sys.argv[2]
    overwrite = sys.argv[3] == "--overwrite"
    json2csv(filename, domain, overwrite)