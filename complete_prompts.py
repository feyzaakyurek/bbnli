import json
import pandas as pd
from transformers import pipeline, GPT2LMHeadModel, GPT2Tokenizer
import argparse
import os
from pretrained_model_list import MODEL_PATH_LIST

def load_prompts(opt):
    pth = os.path.join(opt.prompt_set, "prompts", opt.prompt_domain+"_prompt.json")
    prompts = []
    with open(pth) as f:
        for line in f:
            prompts.append(json.loads(line))
    prompts = prompts[0]

    prompts_df = pd.DataFrame(columns = ["Name", "Group", "Prompt"])
    for group, content in prompts.items():
        for name, prompt_l in content.items():
            for prompt in prompt_l:
                prompts_df.loc[len(prompts_df)] = [name, group, prompt]
    return prompts_df

def get_generations(text_generator, prompts_df, opt):
    num_gens_t = opt.num_gens * len(prompts_df)
    print("Generating total of {} completions.".format(num_gens_t))
    gens = text_generator(prompts_df.Prompt.to_list(),
                          max_length=opt.max_length,
                          do_sample=opt.do_sample,
                          num_return_sequences=opt.num_gens)
    print("Generation completed.")
    gen_df = pd.DataFrame(columns = ["Name", "Group", "Prompt", "Generation"])
    for i,row in prompts_df.iterrows():
        genset = gens[i]
        for gen in genset:
            gen_df.loc[len(gen_df)] = [row['Name'],
                                       row['Group'],
                                       row['Prompt'],
                                       gen['generated_text']]
    gen_df["Generation"] = gen_df['Generation'].str.replace(u'\xa0', u' ')
    return gen_df
    
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser('argument for training')
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--save_path", type=str)
    parser.add_argument("--prompt_set", type=str, default="bold")
    parser.add_argument("--prompt_domain", type=str, default="gender")
    parser.add_argument("--max_length", type=int, default=30)
    parser.add_argument("--do_not_sample", action="store_false", dest="do_sample")
    parser.add_argument("--num_gens", type=int, default=3)
    
    opt = parser.parse_args()
    prompts_df = load_prompts(opt)
    model = GPT2LMHeadModel.from_pretrained(opt.model_path)
    tokenizer = GPT2Tokenizer.from_pretrained(opt.model_path)
    text_generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0)
    gen_df = get_generations(text_generator, prompts_df, opt)
    gen_df.to_csv(opt.save_path)