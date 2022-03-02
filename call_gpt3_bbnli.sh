#!/bin/bash

python complete_prompts.py --prompt_set nlibias \
--prompt_domain race \
--prompt_subtopic black_is_to_impoverished \
--save_path "outputs/nlibias/" \
--model_name gpt3