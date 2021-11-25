from huggingface_hub.inference_api import InferenceApi
from jinja2 import meta
import promptsource.templates
import ipdb
import sys
import pandas as pd
from tqdm import tqdm

API_TOKEN=[el for el in open("hf_key", 'r')][0]
t0_entail_dict={0:2,1:1,2:0}

if __name__ == "__main__":
    model = sys.argv[1] # model name
    pth = sys.argv[2] # csv file
    df = pd.read_csv(pth) 

    # Create inference API
    inference = InferenceApi(repo_id="bigscience/T0pp", token=API_TOKEN)

    # Sets up Jinja environment
    env = promptsource.templates.env

    # Loads templates and iterates over each premise
    template_collection = promptsource.templates.TemplateCollection()
    dataset_templates = template_collection.get_dataset("anli", None)

    new_cols = df.columns.tolist() + ["preds_t0pp", "t0pp_template"]
    new_df = pd.DataFrame(columns=new_cols)

    for template_name in ["GPT-3 style"]:#tqdm(dataset_templates.all_template_names):
        print(template_name)
        template = dataset_templates[template_name]
        
        for ind, row in tqdm(df.iterrows()):
            m = {"premise":row["premise"],
                 "hypothesis":row['hypothesis']}
            inp_only = template.jinja.split(" ||| ")[0]
            inp = env.from_string(inp_only).render(**m)
            try:
                ans = inference(inputs=inp)[0]['generated_text']
            except KeyError:
                ipdb.set_trace()
            ans_choices = template.answer_choices.split(" ||| ")
            if ans in ans_choices:
                pred = t0_entail_dict[ans_choices.index(ans)]
            else:
                pred = -1
            row["preds_t0pp"] = pred
            row["t0pp_template"] = template_name
            new_df.loc[len(new_df)] = row
    new_df.to_csv(pth, index=False)
            
    
