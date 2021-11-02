import json


def read_json(file):
    continue
    
def predict(model, data):
    
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser('argument for training')
    parser.add_argument("--input_json", type=str)
    parser.add_argument("--save_path", type=str)
    parser.add_argument("--prompt_set", type=str, default="bold")
    parser.add_argument("--prompt_domain", type=str, default="gender")
    parser.add_argument("--max_length", type=int, default=30)
    parser.add_argument("--do_not_sample", action="store_false", dest="do_sample")
    parser.add_argument("--num_gens", type=int, default=3)
    
    opt = parser.parse_args()