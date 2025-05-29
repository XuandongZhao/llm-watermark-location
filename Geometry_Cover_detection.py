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
from transformers import AutoModelForCausalLM, AutoTokenizer,LlamaTokenizer
from sentence_transformers import SentenceTransformer
from wm import ( OpenaiDetector, OpenaiDetectorZ, OpenaiGeometryWmDetector, MarylandDetectorZ, MarylandGeometryWmDetector , GPTWatermarkDetector, GeometryWmDetector, Aligator)

def get_args_parser():
    
    parser = argparse.ArgumentParser('Args', add_help=False)

    # parameters for detection
    parser.add_argument('--json_path', type=str, default = "./detect_file.jsonl")
    parser.add_argument('--output_json_path', type=str, default = "output.jsonl")
    parser.add_argument('--text_key', type=str, default='text',help='key to access text in json dict')
    parser.add_argument('--tokenizer_dir', type=str, default="mistralai/Mistral-7B-Instruct-v0.2")

    # watermark parameters
    parser.add_argument('--method', type=str, default='marylandz',
                        help='watermark detection method')
    parser.add_argument('--seeding', type=str, default='hash', 
                        help='seeding method for rng key generation as introduced in https://github.com/jwkirchenbauer/lm-watermarking')
    parser.add_argument('--ngram', type=int, default=2, 
                        help='watermark context width for rng key generation')
    parser.add_argument('--n_positive', type=int, default=500, 
                        help='number of positive samples to evaluate, if None, take all texts')
    parser.add_argument('--gamma', type=float, default=0.25, 
                        help='gamma for maryland: proportion of greenlist tokens')
    parser.add_argument('--hash_key', type=int, default=35317, 
                        help='hash key for rng key generation')
    parser.add_argument('--scoring_method', type=str, default='none', 
                        help='method for scoring. choose between: \
                        none (score every tokens), v1 (score token when wm context is unique), \
                        v2 (score token when {wm context + token} is unique')
    parser.add_argument('--payload', type=int, default=0, 
                        help='message')
    parser.add_argument('--payload_max', type=int, default=4, 
                        help='maximal message')
    parser.add_argument('--delta', type=float, default=2.0, 
                        help='delta for maryland (useless for detection)')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--seed', type=int, default=0)

    return parser

def load_results(json_path: str, nsamples: int=None, text_key: str='result') -> List[str]:
    with open(json_path, "r") as f:
        if json_path.endswith('.json'):
            prompts = json.loads(f.read())
        else:
            prompts = [json.loads(line) for line in f.readlines()] 
    new_prompts = [o[text_key] for o in prompts]
    new_prompts = new_prompts[0:nsamples]
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

    # hyperparameters
    param_grid = {
        'start_length': [ 32  ],
        'voting_score_threshold': [1] ,
        'pthreshold': [0.001,0.01,0.05] 
    }

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # build tokenizer
    if "llama" in args.tokenizer_dir:
        tokenizer = LlamaTokenizer.from_pretrained(args.tokenizer_dir,torch_dtype=torch.float16)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir)

    if args.method == 'unigram':
        detector = GeometryWmDetector(fraction=args.gamma,
                                    strength=args.delta,
                                    vocab_size=tokenizer.vocab_size,
                                    watermark_key=args.seed)
        
    # load results and (optional) do splits
    results = load_results(json_path=args.json_path, text_key=args.text_key, nsamples=args.n_positive)
    wm_id, wm_begin, wm_end = load_data(json_path=args.json_path, nsamples=args.n_positive)


    # with open(args.output_json_path,"w") as outf:
    for length in param_grid['start_length']:
            for Vscore in param_grid['voting_score_threshold']:
                for pvalue in param_grid['pthreshold']:

                    if args.method == 'marylandz':
                        geometry_detector = MarylandGeometryWmDetector(length,Vscore,pvalue,tokenizer, args.ngram, args.seed, args.seeding, args.hash_key, gamma=args.gamma, delta=args.delta)
                    elif args.method == 'openaiz':
                        geometry_detector = OpenaiGeometryWmDetector(length,Vscore,pvalue,tokenizer, args.ngram, args.seed, args.seeding, args.hash_key)
                    
                    zero_count = 0
                    general_detect_count = 0
                    partial_detect_count = 0

                    for ii, text in tqdm.tqdm(enumerate(results), total=len(results)):
                        gen_tokens = tokenizer(text, add_special_tokens=False)["input_ids"] 
                        if args.method == 'marylandz' or args.method == 'openaiz':
                            wm_tokens, IOU_score,dtpvalue=geometry_detector.geometry_detect(gen_tokens,wm_begin[ii],wm_end[ii],args.payload_max)
                            

                            # log_stat = {"General_Pvalue" : pvalues[0],
                            #          "Partial_Pvalue" : dtpvalue,
                            #          "Detected_len" : len(wm_tokens),
                            #          "IOU_score" : IOU_score
                            #         }
                        
                        elif args.method == 'unigram' :
                            general_res, z_score = detector.dynamic_threshold(gen_tokens, pvalue)
                            wm_tokens ,IOU_score, wm_zscore= detector.geometry_detect(gen_tokens, 0 , wm_begin[ii], wm_end[ii], pvalue)    
                        
                            # log_stat = {"General_zscore" : z_score,
                            #          "Partial_zscore" : float(wm_zscore),
                            #          "Detected_len" : len(wm_tokens),
                            #          "IOU_score" : IOU_score
                            #         }
                            # if general_res: 
                            #     general_detect_count += 1
                           

                        if len(wm_tokens) == 0 :
                            zero_count += 1
                        elif len(wm_tokens)>=20 :
                            partial_detect_count += 1
                        
                        # outf.write( json.dumps(log_stat) + '\n' )
                    print(f"-----------------Pvalue = {pvalue}-------------------")
                    print("Zero_rate ",zero_count/len(results))
                    # print("General_detect_rate ", general_detect_count/len(results))
                    print("Partial_detect_rate ", partial_detect_count/len(results))
                    print("Total_len:",len(results))
    

if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)
