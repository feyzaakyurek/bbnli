# python hf_inference_api.py --path BBQ/templates/new_templates\ -\ Gender_identity_pro.csv --dataset bbq --prompt_form qa

from huggingface_hub.inference_api import InferenceApi
from jinja2 import meta
import promptsource.templates
import ipdb
import pandas as pd
import argparse
from tqdm import tqdm

API_TOKEN=[el for el in open("hf_key", 'r')][0]
t0_entail_dict={0:2,1:1,2:0}
nli_anli_valid = ["GPT-3 style",
                  "MNLI crowdsource"]
qa_quoref_valid = ["Answer Question Given Context",
                   "Given Context Answer Question"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser('argument for sent and tox')
    # parser.add_argument("--model", type=str, default="t0pp")
    parser.add_argument("--path", type=str, help="input csv file")
    parser.add_argument("--dataset", type=str, default="nli_inhouse")
    parser.add_argument("--prompt_form", type=str, default="nli")
    opt = parser.parse_args()

    # model = opt.model
    pth = opt.path
    dataset = opt.dataset
    prompt_form = opt.prompt_form

    # Read input
    df = pd.read_csv(pth) 

    # Create inference API
    inference = InferenceApi(repo_id="bigscience/T0pp", token=API_TOKEN)

    # Sets up Jinja environment
    env = promptsource.templates.env

    # Loads templates and iterates over each premise
    template_collection = promptsource.templates.TemplateCollection()
    if prompt_form == "nli":
        dataset_templates = template_collection.get_dataset("anli", None)
    else:
        dataset_templates = template_collection.get_dataset("quoref", None)

    new_cols = df.columns.tolist() + ["preds_t0pp"+prompt_form, "t0pp_template"+prompt_form]
    new_df = pd.DataFrame(columns=new_cols)
    template_names = dataset_templates.all_template_names
    template_names = nli_anli_valid if prompt_form == "nli" else qa_quoref_valid


    for template_name in tqdm(template_names):
        print(template_name)
        template = dataset_templates[template_name]
        
        for ind, row in tqdm(df.iterrows()):
            if prompt_form == "nli":
                if dataset == "nli_inhouse":
                    m = {"premise": row["premise"],
                        "hypothesis": row['hypothesis']}
                else:  # BBQ
                    premise = row["Ambiguous_Context"] + " " + row["NLI_plug_alternative"]
                    hypothesis = row["Question_negative_stereotype_nli"]
                    m = {"premise": premise,
                        "hypothesis": hypothesis}
            else:  # QA
                if dataset == "nli_inhouse":
                    raise NotImplementedError()
                else:
                    context = row["Ambiguous_Context"] + " " + row["NLI_plug_alternative"]
                    question = row["Question_negative_stereotype"]
                    m = {"question": question,
                         "context": context }

            inp_only = template.jinja.split(" |||")[0]
            print(inp_only)
            print(m)
            inp = env.from_string(inp_only).render(**m)
            try:
                ans = inference(inputs=inp)[0]['generated_text']
                print(ans)
            except KeyError:
                ipdb.set_trace()
            
            if prompt_form == "nli":
                ans_choices = template.answer_choices.split(" ||| ")
                if ans in ans_choices:
                    pred = t0_entail_dict[ans_choices.index(ans)]
                else:
                    pred = -1
            else:
                pred = ans
            row["preds_t0pp"+prompt_form] = pred
            row["t0pp_template"+prompt_form] = template_name
            new_df.loc[len(new_df)] = row

    new_df.to_csv(pth, index=False)
            
    
