import json
import pandas as pd
from jinja2.nativetypes import NativeEnvironment
import os
import sys

if __name__ == "__main__":
    filename = sys.argv[1] #"data/nli/gender/man_is_to_programmer.json"
    print(filename)
    with open(filename, "r") as f:
        inp = json.load(f)
    env = NativeEnvironment()
    
    group1 = inp["GROUP1"][0]
    group2 = inp["GROUP2"][0]
    name = inp["name"].replace(" ", "_")
    df = pd.DataFrame(columns=["domain",
                               "premise_id",
                               "premise",
                               "hypothesis_type",
                               "hypothesis",
                               "true_label",
                               "bias_label",
                               "reference"
                               ])
    for ind,pre in enumerate(inp['premise']):
    #     for job in inp['MJOB_list']:
        pre = env.from_string(pre).render(MJOB="software engineering", GROUP1=group1, GROUP2=group2)
        for test_hyp,test_hyp_l in inp['test_hypothesis']:
            test_hyp = env.from_string(test_hyp).render(MJOB="software_engineering", GROUP1=group1, GROUP2=group2)
            df.loc[len(df)] = [inp['domain'],name+"_premise"+str(ind), pre, "test", test_hyp, test_hyp_l, -1, inp['reference'][ind]]
        for hyp, label, bias_label in inp['bias_hypothesis_stereotypical']:
            hyp = env.from_string(hyp).render(MJOB="software engineering", GROUP1=group1, GROUP2=group2)
            df.loc[len(df)] = [inp['domain'],name+"_premise"+str(ind), pre, "stereotypical", hyp, label, bias_label, inp['reference'][ind]]
        for hyp, label, bias_label in inp['bias_hypothesis_anti-stereotypical']:
            hyp = env.from_string(hyp).render(MJOB="software engineering", GROUP1=group1, GROUP2=group2)
            df.loc[len(df)] = [inp['domain'],name+"_premise"+str(ind), pre, "anti-stereotypical", hyp, label, bias_label, inp['reference'][ind]]
            
    head, tail = os.path.split(filename)
    pth = os.path.join(head, tail.replace("json", "csv"))
    df.to_csv(pth)