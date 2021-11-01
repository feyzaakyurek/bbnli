# pose sequence as a NLI premise and label as a hypothesis
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import pandas as pd
import ipdb

if __name__ == "__main__":
    device = "cpu"
    nli_model = AutoModelForSequenceClassification.from_pretrained('facebook/bart-large-mnli', device=device)
    tokenizer = AutoTokenizer.from_pretrained('facebook/bart-large-mnli')

    df = pd.read_csv("data/nli/gender/man_is_to_programmer.csv")
    
    premise = df['premise']
    hypothesis = df['hypothesis']

    # run through model pre-trained on MNLI
    x = tokenizer.encode(premise, hypothesis, return_tensors='pt',
                         truncation_strategy='only_first')
    logits = nli_model(x.to(device))#[0]

    # we throw away "neutral" (dim 1) and take the probability of
    # "entailment" (2) as the probability of the label being true 
    ipdb.set_trace()
    entail_contradiction_logits = logits[:,[0,2]]
    probs = entail_contradiction_logits.softmax(dim=1)
    prob_label_is_true = probs[:,1]