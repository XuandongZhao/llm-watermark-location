import argparse
from typing import List
import os
import json
from math import exp

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

import tqdm
import pandas as pd
import numpy as np

import torch
from peft import PeftModel    
from transformers import AutoModelForCausalLM, AutoTokenizer,LlamaTokenizer
from sentence_transformers import SentenceTransformer
from wm import ( OpenaiDetector, OpenaiDetectorZ, OpenaiAligator, MarylandAligator, OpenaiGeometryWmDetector, Aligator)

def get_args_parser():
    parser = argparse.ArgumentParser('Args', add_help=False)

    # exp parameters
    parser.add_argument('--json_path', type=str, default = "./Openai_Sci_p001_combine.jsonl")
    parser.add_argument('--text_key', type=str, default='text',help='key to access text in json dict')
    parser.add_argument('--tokenizer_dir', type=str, default="mistralai/Mistral-7B-Instruct-v0.2")
    parser.add_argument('--n_positive', type=int, default=500, 
                        help='number of positive samples to evaluate, if None, take all texts')

    # watermark parameters
    parser.add_argument('--method', type=str, default='marylandz',
                        help='watermark detection method')
    parser.add_argument('--seeding', type=str, default='hash', 
                        help='seeding method for rng key generation as introduced in https://github.com/jwkirchenbauer/lm-watermarking')
    parser.add_argument('--ngram', type=int, default=2, 
                        help='watermark context width for rng key generation')
    parser.add_argument('--hash_key', type=int, default=35317, 
                        help='hash key for rng key generation')
    parser.add_argument('--scoring_method', type=str, default='none', 
                        help='method for scoring. choose between: \
                        none (score every tokens), v1 (score token when wm context is unique), \
                        v2 (score token when {wm context + token} is unique')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--Aaronson_thres', type=float, default=1.3)
    parser.add_argument('--RedGreen_thres', type=float, default=0.65)
    parser.add_argument('--gamma', type=float, default=0.25)
    parser.add_argument('--delta', type=float, default=2)
    # multibit
    parser.add_argument('--payload', type=int, default=0, 
                        help='message')
    parser.add_argument('--payload_max', type=int, default=4, 
                        help='maximal message')

    return parser

def load_results(json_path: str, nsamples: int=None, text_key: str='result') -> List[str]:
    with open(json_path, "r") as f:
        if json_path.endswith('.json'):
            prompts = json.loads(f.read())
        else:
            prompts = [json.loads(line) for line in f.readlines()] 
    new_prompts = [o[text_key] for o in prompts]
    new_prompts = new_prompts[:nsamples]
    return new_prompts

def load_data(json_path: str, nsamples: int=None, index_key: str="text", gt_begin:str = "start_wm_index",gt_end:str = "end_wm_index") -> List[str]:
    with open(json_path, "r") as f:
        if json_path.endswith('.json'):
            prompts = json.loads(f.read())
        else:
            prompts = [json.loads(line) for line in f.readlines()] 
    index = [o[index_key] for o in prompts]
    begin = [o[gt_begin] for o in prompts]
    end = [o[gt_end] for o in prompts]
    index = index[:nsamples]
    begin = begin[:nsamples]
    end = end[:nsamples]
    return index,begin,end

def main(args):

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # build tokenizer
    if "llama" in args.tokenizer_dir:
        tokenizer = LlamaTokenizer.from_pretrained(args.tokenizer_dir,torch_dtype=torch.float16)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir)

    if args.method == "openaiz":
        AligatorDetector = OpenaiAligator(args.Aaronson_thres, tokenizer, args.ngram, args.seed, args.seeding, args.hash_key)
    elif args.method == "marylandz":
        AligatorDetector = MarylandAligator(args.RedGreen_thres, tokenizer, args.ngram, args.seed, args.seeding, args.hash_key, args.gamma, args.delta)
    elif args.method == "unigram":
        AligatorDetector = Aligator(fraction=args.gamma,
                                    strength=args.delta,
                                    vocab_size=tokenizer.vocab_size,
                                    watermark_key=args.seed)
        


    # load results and (optional) do splits
    results = load_results(json_path=args.json_path, text_key=args.text_key, nsamples=args.n_positive)
    wm_id, wm_begin, wm_end = load_data(json_path=args.json_path, nsamples=args.n_positive)
    
    IOU_scores = []
    for ii, text in tqdm.tqdm(enumerate(results), total=len(results)):
            
            if args.method == 'marylandz' or args.method == 'openaiz':
                alig , IOU_score , detect_res = AligatorDetector.detect(text, wm_begin[ii], wm_end[ii],args.payload_max)
            elif args.method == 'unigram':
                gen_tokens = tokenizer(text, add_special_tokens=False)["input_ids"]
                alig , IOU_score , detect_res = AligatorDetector.detect(gen_tokens, args.RedGreen_thres,  wm_begin[ii],  wm_end[ii])
            print("idx = ",ii, IOU_score)
            IOU_scores.append(IOU_score)

    IOU_average = sum(IOU_scores) / len(IOU_scores)
    print("IOU_average: ",IOU_average)

if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)
