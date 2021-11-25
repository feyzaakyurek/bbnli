import json
import pandas as pd
from transformers import pipeline, GPT2LMHeadModel, GPT2Tokenizer
import argparse
import os
import openai
from pretrained_model_list import MODEL_PATH_LIST


def clean_up_tokenization(out_string: str) -> str:
    """
    Clean up a list of simple English tokenization artifacts like spaces before punctuations and abbreviated forms.
    Args:
        out_string (:obj:`str`): The text to clean up.
    Returns:
        :obj:`str`: The cleaned-up string.
    """
    out_string = (
        out_string.replace(" .", ".")
        .replace(" ?", "?")
        .replace(" !", "!")
        .replace(" ,", ",")
        .replace(" ' ", "'")
        .replace(" n't", "n't")
        .replace(" 'm", "'m")
        .replace(" 's", "'s")
        .replace(" 've", "'ve")
        .replace(" 're", "'re")
        .replace("\n\n", " ")
        .replace("\n", " ")
        .replace("\r", " ")
    )
    return out_string

def load_prompts(opt):
    pth = os.path.join("data", opt.prompt_set, "prompts", opt.prompt_domain+"_prompt.json")
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

def get_generations_gpt2(prompts_df, opt):
    model = GPT2LMHeadModel.from_pretrained(opt.model_path)
    tokenizer = GPT2Tokenizer.from_pretrained(opt.model_path)
    text_generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0)
        
    num_gens_t = opt.num_gens * len(prompts_df)
    print("Generating total of {} completions.".format(num_gens_t))
    gens = text_generator(prompts_df.Prompt.to_list(),
                          max_length=opt.max_length,
                          do_sample=opt.do_sample,
                          num_return_sequences=opt.num_gens,
                          clean_up_tokenization_spaces=True)
    
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


def get_generations_gpt3(prompts_df, opt):
    def chunks(prompts_df, n):
        """Yield successive n-sized chunks from lst."""
        for i in range(0, len(prompts_df), n):
            yield prompts_df.iloc[i:min(i + n, len(prompts_df)),:]

    openai.api_key = [el for el in open("openai_key", 'r')][0]
    gen_df = pd.DataFrame(columns = ["Name", "Group", "Prompt", "Generation"])

    for chunk in list(chunks(prompts_df, opt.batch_size)):
        # create a completion
        lst = [el.strip(" ") for el in chunk.Prompt.to_list()]
        completion = openai.Completion.create(engine="curie",
                                              prompt=lst,
                                              max_tokens=opt.max_length,
                                              top_p=0.9,
                                              n=opt.num_gens)
        count = 0
        for i,row in chunk.iterrows():
            for j in range(opt.num_gens):
                cln = clean_up_tokenization(completion.choices[count].text)
                gen_df.loc[len(gen_df)] = [row['Name'],
                                           row['Group'],
                                           row['Prompt'],
                                           cln]
                count += 1

    gen_df["Generation"] = gen_df['Generation'].str.replace(u'\xa0', u' ')
    return gen_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser('argument for training')
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--save_path", type=str)
    parser.add_argument("--prompt_set", type=str, default="bold")
    parser.add_argument("--prompt_domain", type=str, default="gender")
    parser.add_argument("--max_length", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--do_not_sample", action="store_false", dest="do_sample")
    parser.add_argument("--num_gens", type=int, default=3)

    opt = parser.parse_args()
    os.makedirs(opt.save_path, exist_ok=True)

    if not opt.do_sample:
        assert opt.num_gens == 1
        
    prompts_df = load_prompts(opt)
    if opt.model_name == "gpt2":
        gen_df = get_generations_gpt2(prompts_df, opt)
    elif opt.model_name == "gpt3":
        gen_df = get_generations_gpt3(prompts_df, opt)
    else:
        raise ValueError(f"{opt.model_name} is not known.")
    pth = os.path.join(opt.save_path, "gens.csv")
    gen_df.to_csv(pth)