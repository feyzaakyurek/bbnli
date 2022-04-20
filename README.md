# On Measuring Social Biases in Prompt-Based Multi-Task Learning

Code repository for the following paper where we test whether zero-shot generalization capacity of T0 models into novel language tasks results in amplified social biases.

On Measuring Social Biases in Prompt-Based Multi-Task Learning, Afra Feyza Akyürek, Sejin Paik, Muhammed Yusuf Kocyigit, Seda Akbiyik, Şerife Leman Runyun, Derry Tanti Wijaya. Findings of NAACL 2022. [[Link]](https://openreview.net/forum?id=SAgxG-jZHZq)

![](bias_teaser.png)

## Installation & Running

After cloning the repository and changing directory, please consider using `sh setup.sh` to create a conda environment and install required packages. You can then run `python call_bbnli_inference_bias.py` for BBNLI and `python bbq_inference_bias.py` for BBQ experiments. For BBNLI, the script first converts the set of premises and hypotheses into a csv file while cross-pairing every premise with every hypothesis. It then continues to sample generations from the model using HugginFace Inference API. For this stage to work, you need to a file named `hf_key` under the repository containing only your API key. 

There are a few flags at the beginning of each script, when `overwrite_csv=False` the script skips creating csv files and goes directly into generating samples. When `skip_inference=True`, the script doesn't generate completions (assuming you already have them in the respective csv files), instead solely computes bias scores. 

## BBNLI Dataset

We release a natural language inference benchmark for measuring biases covering gender, race and religion domains. You may find the dataset under `data/bbnli`.

## Contacts
Feel free to reach out to with questions, or open issues.

Afra Feyza Akyürek (akyurek@bu.edu)

## Cite

If you are using code or data from this repository, please cite the following work:

```
@inproceedings{akyurek2022promptbased,
  title={On Measuring Social Biases in Prompt-Based Multi-Task Learning},
  author={Aky{\"u}rek, Afra Feyza and Paik, Sejin and Kocyigit, Muhammed Yusuf and Akbiyik, Seda and Runyun, Serife Leman and Wijaya, Derry},
  booktitle={Findings of NAACL},
  year={2022}
}
```




