import json
import pandas as pd
import pdb

def read_jsonl(pth):
    train_data = []
    with open(pth) as f:
        for line in f:
            train_data.append(json.loads(line))

    df_inter = pd.DataFrame(train_data)
    df_inter['text'] = df_inter['text'].str.replace('\n', ' ')
    return df_inter
    
if __name__=="__main__":
#     for cat in ["combined"]:
    MAIN="/home/gridsan/akyurek/git/bias-lm"
    TRAIN_FILE=MAIN+f"/gpt-2-output-dataset/data/webtext.train.jsonl"
    VALID_FILE=MAIN+f"/gpt-2-output-dataset/data/webtext.valid.jsonl"

    TRAIN_FILE_=MAIN+f"/gpt-2-output-dataset/data/webtext.train.txt"
    VALID_FILE_=MAIN+f"/gpt-2-output-dataset/data/webtext.valid.txt"

    train_d = read_jsonl(TRAIN_FILE)
    valid_d = read_jsonl(VALID_FILE)

    with open(TRAIN_FILE_, 'w') as f:
        for line in train_d['text']:
            f.write("%s\n" % line)

    with open(VALID_FILE_, 'w') as f:
        for line in valid_d['text']:
            f.write("%s\n" % line)
        
#         read_jsonl(TRAIN_FILE).to_csv(TRAIN_FILE_)
#         read_jsonl(VALID_FILE).to_csv(VALID_FILE_)