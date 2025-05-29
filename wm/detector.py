# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import hashlib
from typing import List, Dict, Callable

import numpy as np
from scipy import special
from scipy.optimize import fminbound
import random
import cpp_src.aligator as aligator
from scipy.stats import norm
import torch
from transformers import LlamaTokenizer, LlamaForCausalLM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class WmDetector():
    '''
     Adapted from https://github.com/facebookresearch/three_bricks
    '''
    def __init__(self, 
            tokenizer: LlamaTokenizer, 
            ngram: int = 1,
            seed: int = 0,
            seeding: str = 'hash',
            salt_key: int = 35317
        ):
        # model config
        self.tokenizer = tokenizer
        self.vocab_size = self.tokenizer.vocab_size
        # watermark config
        self.ngram = ngram
        self.salt_key = salt_key
        self.seed = seed
        self.hashtable = torch.randperm(1000003)
        self.seeding = seeding 
        self.rng = torch.Generator()
        self.rng.manual_seed(self.seed)

    def hashint(self, integer_tensor: torch.LongTensor) -> torch.LongTensor:
        """Adapted from https://github.com/jwkirchenbauer/lm-watermarking"""
        return self.hashtable[integer_tensor.cpu() % len(self.hashtable)] 
    
    def get_seed_rng(self, input_ids: List[int]) -> int:
        """
        Seed RNG with hash of input_ids.
        Adapted from https://github.com/jwkirchenbauer/lm-watermarking
        """
        if self.seeding == 'hash':
            seed = self.seed
            for i in input_ids:
                seed = (seed * self.salt_key + i) % (2 ** 64 - 1)
        elif self.seeding == 'additive':
            seed = self.salt_key * torch.sum(input_ids)
            seed = self.hashint(seed)
        elif self.seeding == 'skip':
            seed = self.salt_key * input_ids[0]
            seed = self.hashint(seed)
        elif self.seeding == 'min':
            seed = self.hashint(self.salt_key * input_ids)
            seed = torch.min(seed)
        return int(seed)

    def aggregate_scores(self, scores: List[List[np.array]], aggregation: str = 'mean') -> List[float]:
        """Aggregate scores along a text."""
        scores = np.asarray(scores)
        if aggregation == 'sum':
           return [ss.sum(axis=0) for ss in scores]
        elif aggregation == 'mean':
            return [ss.mean(axis=0) if ss.shape[0]!=0 else np.ones(shape=(self.vocab_size)) for ss in scores]
        elif aggregation == 'max':
            return [ss.max(axis=0) for ss in scores]
        else:
             raise ValueError(f'Aggregation {aggregation} not supported.')

    def get_scores_by_t(
        self, 
        texts, 
        scoring_method: str="none",
        ntoks_max: int = None,
        payload_max: int = 0
    ) -> List[List[np.array]]:
        """
        Get score increment for each token in list of texts.
        Args:
            texts: list of texts
            scoring_method: 
                'none': score all ngrams
                'v1': only score tokens for which wm window is unique
                'v2': only score unique {wm window+tok} is unique
            ntoks_max: maximum number of tokens
            payload_max: maximum number of messages 
        Output:
            score_lists: list of [np array of score increments for every token and payload] for each text
        """
        bsz = len(texts)
        # tokens_id = [self.tokenizer.encode(x, add_special_tokens=False) for x in texts]

        if ntoks_max is not None:
            texts = [x[:ntoks_max] for x in texts]
        score_lists = []
        for ii in range(bsz):
            total_len = len(texts[ii])
            start_pos = self.ngram +1
            rts = []
            seen_ntuples = set()
            for cur_pos in range(start_pos, total_len):
                ngram_tokens = texts[ii][cur_pos-self.ngram:cur_pos] # h
                if scoring_method == 'v1':
                    tup_for_unique = tuple(ngram_tokens)
                    if tup_for_unique in seen_ntuples:
                        continue
                    seen_ntuples.add(tup_for_unique)
                elif scoring_method == 'v2':
                    tup_for_unique = tuple(ngram_tokens + texts[ii][cur_pos:cur_pos+1])
                    if tup_for_unique in seen_ntuples:
                        continue
                    seen_ntuples.add(tup_for_unique)
                rt = self.score_tok(ngram_tokens, int(texts[ii][cur_pos])) 
                rt = rt.numpy()[:payload_max+1]
                rts.append(rt)
        
            score_lists.append(rts)
        
        return score_lists

    def get_scores_by_text(
        self, 
        texts: List[str], 
        scoring_method: str="none",
        ntoks_max: int = None,
        payload_max: int = 0
    ) -> List[List[np.array]]:
        """
        Get score increment for each token in list of texts.
        Args:
            texts: list of texts
            scoring_method: 
                'none': score all ngrams
                'v1': only score tokens for which wm window is unique
                'v2': only score unique {wm window+tok} is unique
            ntoks_max: maximum number of tokens
            payload_max: maximum number of messages 
        Output:
            score_lists: list of [np array of score increments for every token and payload] for each text
        """
        bsz = len(texts)
        tokens_id = [self.tokenizer.encode(x, add_special_tokens=False) for x in texts]

        if ntoks_max is not None:
            tokens_id = [x[:ntoks_max] for x in tokens_id]
        score_lists = []
        for ii in range(bsz):
            total_len = len(tokens_id[ii])
            start_pos = self.ngram +1
            rts = []
            seen_ntuples = set()
            for cur_pos in range(start_pos, total_len):
                ngram_tokens = tokens_id[ii][cur_pos-self.ngram:cur_pos] # h
                if scoring_method == 'v1':
                    tup_for_unique = tuple(ngram_tokens)
                    if tup_for_unique in seen_ntuples:
                        continue
                    seen_ntuples.add(tup_for_unique)
                elif scoring_method == 'v2':
                    tup_for_unique = tuple(ngram_tokens + tokens_id[ii][cur_pos:cur_pos+1])
                    if tup_for_unique in seen_ntuples:
                        continue
                    seen_ntuples.add(tup_for_unique)
                rt = self.score_tok(ngram_tokens, tokens_id[ii][cur_pos]) 
                rt = rt.numpy()[:payload_max+1]
                
                rts.append(rt)
            
            score_lists.append(rts)
        
        return score_lists
    
    def get_pvalues(
            self, 
            scores: List[np.array], 
            eps: float=1e-200
        ) -> np.array:
        """
        Get p-value for each text.
        Args:
            score_lists: list of [list of score increments for each token] for each text
        Output:
            pvalues: np array of p-values for each text and payload
        """
        pvalues = []
        scores = np.asarray(scores) # bsz x ntoks x payload_max

        for ss in scores:
            ntoks = ss.shape[0]
            scores_by_payload = ss.sum(axis=0) if ntoks!=0 else np.zeros(shape=ss.shape[-1]) # payload_max
            pvalues_by_payload = [self.get_pvalue(score, ntoks, eps=eps) for score in scores_by_payload]
            pvalues.append(pvalues_by_payload)
        return np.asarray(pvalues) # bsz x payload_max

    def get_pvalues_by_t(self, scores: List[float]) -> List[float]:
        """Get p-value for each text."""
        pvalues = []
        cum_score = 0
        cum_toks = 0
        for ss in scores:
            cum_score += ss
            cum_toks += 1
            pvalue = self.get_pvalue(cum_score, cum_toks)
            pvalues.append(pvalue)
        return pvalues
    
    def score_tok(self, ngram_tokens: List[int], token_id: int):
        """ for each token in the text, compute the score increment """
        raise NotImplementedError
    
    def get_pvalue(self, score: float, ntoks: int, eps: float):
        """ compute the p-value for a couple of score and number of tokens """
        raise NotImplementedError

class MarylandDetector(WmDetector):
    '''
     Adapted from https://github.com/facebookresearch/three_bricks
    '''
    def __init__(self, 
            tokenizer: LlamaTokenizer,
            ngram: int = 1,
            seed: int = 0,
            seeding: str = 'hash',
            salt_key: int = 35317,
            gamma: float = 0.5, 
            delta: float = 1.0, 
            **kwargs):
        super().__init__(tokenizer, ngram, seed, seeding, salt_key, **kwargs)
        self.gamma = gamma
        self.delta = delta
    
    def score_tok(self, ngram_tokens, token_id):
        """ 
        score_t = 1 if token_id in greenlist else 0 
        The last line shifts the scores by token_id. 
        ex: scores[0] = 1 if token_id in greenlist else 0
            scores[1] = 1 if token_id in (greenlist shifted of 1) else 0
            ...
        The score for each payload will be given by scores[payload]
        """
        seed = self.get_seed_rng(ngram_tokens)
        self.rng.manual_seed(seed)
        scores = torch.zeros(self.vocab_size)
        vocab_permutation = torch.randperm(self.vocab_size, generator=self.rng)
        greenlist = vocab_permutation[:int(self.gamma * self.vocab_size)] # gamma * n toks in the greenlist
        scores[greenlist] = 1 
        return scores.roll(-token_id) 
                
    def get_pvalue(self, score: int, ntoks: int, eps: float):
        """ from cdf of a binomial distribution """
        pvalue = special.betainc(score, 1 + ntoks - score, self.gamma)
        return max(pvalue, eps)

class MarylandDetectorZ(WmDetector):
    '''
     Adapted from https://github.com/facebookresearch/three_bricks
    '''
    def __init__(self, 
            tokenizer: LlamaTokenizer,
            ngram: int = 1,
            seed: int = 0,
            seeding: str = 'hash',
            salt_key: int = 35317,
            gamma: float = 0.5, 
            delta: float = 1.0, 
            **kwargs):
        super().__init__(tokenizer, ngram, seed, seeding, salt_key, **kwargs)
        self.gamma = gamma
        self.delta = delta
    
    def score_tok(self, ngram_tokens, token_id):
        """ same as MarylandDetector but using zscore """
        seed = self.get_seed_rng(ngram_tokens)
        self.rng.manual_seed(seed)
        scores = torch.zeros(self.vocab_size)
        vocab_permutation = torch.randperm(self.vocab_size, generator=self.rng)
        greenlist = vocab_permutation[:int(self.gamma * self.vocab_size)] # gamma * n
        scores[greenlist] = 1
        return scores.roll(-token_id)
                
    def get_pvalue(self, score: int, ntoks: int, eps: float):
        """ from cdf of a normal distribution """
        zscore = (score - self.gamma * ntoks) / np.sqrt(self.gamma * (1 - self.gamma) * ntoks)
        pvalue = 0.5 * special.erfc(zscore / np.sqrt(2))
        return max(pvalue, eps)

    def get_zscore(self, score : float, ntoks : int):
        zscore = (score - self.gamma) / np.sqrt(self.gamma * (1 - self.gamma) /ntoks)
        return zscore

class MarylandGeometryWmDetector(MarylandDetectorZ):
    """
    Class for detecting watermarks in long text.

    Args:
        length: Geometry cover starts from this length
        Vscore: Voting_score_threshold. Normarlly set to 0, meaning once detected in one segment, we recognize it as watermarked.
        pvalue: The P_value threshold for one segment.
        fraction: The fraction of the distribution to be green-listed.
        strength: The strength of the green-listing. Higher values result in higher logit scores for green-listed tokens.
        vocab_size: The size of the vocabulary.
        watermark_key: The random seed for the green-listing.
    """
    def __init__(self,length,Vscore,pvalue,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.length = length
        self.Vscore = Vscore
        self.pvalue = pvalue
    
    def geometry_cover(self,sequence:List[float],payload_max)->List[List[float]]:
            length_seq = len(sequence)
            geometry_scores = []
            i = self.length
            scores_no_aggreg = self.get_scores_by_t([sequence], scoring_method="none", payload_max=payload_max)
            while i<length_seq - self.ngram:
                i_score = []
                i *= 2
                for j in range(0,length_seq - self.ngram,i):
                    total_len = length_seq-j if j+i > length_seq else i
                    if total_len<64 :
                        break
                    # scores_no_aggreg = self.get_scores_by_t([sequence[j:j+total_len]], scoring_method="none", payload_max=payload_max)
                    # scores = self.aggregate_scores(scores_no_aggreg) # p 1
                    pvalues = self.get_pvalues([scores_no_aggreg[0][j:j+total_len]])
                    
                    if payload_max:
                        # decode payload and adjust pvalues
                        payloads = np.argmin(pvalues, axis=1).tolist()
                        pvalues = pvalues[:,payloads][0].tolist()
                        M = payload_max+1
                        pvalues = [(1 - (1 - pvalue)**M) if pvalue > min(1 / M, 1e-5) else M * pvalue for pvalue in pvalues]
                    else:
                        payloads = [ 0 ] * len(pvalues)
                        pvalues = pvalues[:,0].tolist()
                        
                    i_score.append(pvalues[0])
                geometry_scores.append(i_score)

            return geometry_scores
    
    def voting_score(self,scores:List[List[float]],sequence:List[int])->List[float]:
        length_seq = len(sequence)
        voting_res =[0.0] * length_seq
        voting_res = np.array(voting_res)
        scores_idx = 0
        i = self.length
        offset_idx = 0
        while i<length_seq - self.ngram:
            i *= 2
            tmp_index = len(scores[scores_idx])
            tokens_idx = 0
            for j in range(0,tmp_index):
                if scores[scores_idx][j] < self.pvalue: 
                    if tokens_idx+i<=length_seq:
                        voting_res[tokens_idx:tokens_idx+i] += 1  
                    else:
                        voting_res[tokens_idx:] += 1
                tokens_idx += i
            offset_idx +=1
            scores_idx += 1 
        return voting_res.tolist() ,scores 
    
    def geometry_detect_without_IOU(self, sequence: List[int],payload_max:int) -> float:
        """Detect the watermark in a sequence of tokens and return the z value.
           Do not calculate IOU.
        """
        detect_res,scores = self.voting_score(self.geometry_cover(sequence,payload_max),sequence)
        detect_res = np.array(detect_res)
        sequence = np.array(sequence)

        # voting_score > threshold
        wm_res = np.where(detect_res >= self.Vscore,1,0)
        wm_tokens = sequence[wm_res == 1]

        wm_detect = np.where(detect_res >= self.Vscore)[0]

        detect_set = set(wm_detect)
        if wm_detect.shape[0] < 32:
            return wm_tokens,None 
        
        scores_no_aggreg = self.get_scores_by_t([wm_tokens], scoring_method="none", payload_max=payload_max)
        scores = self.aggregate_scores(scores_no_aggreg) 
        pvalues = self.get_pvalues(scores_no_aggreg)
        if payload_max:
            # decode payload and adjust pvalues
            payloads = np.argmin(pvalues, axis=1).tolist()
            pvalues = pvalues[:,payloads][0].tolist()
            scores = [float(s[payload]) for s,payload in zip(scores,payloads)]
            M = payload_max+1
            pvalues = [(1 - (1 - pvalue)**M) if pvalue > min(1 / M, 1e-5) else M * pvalue for pvalue in pvalues]
        else:
            payloads = [ 0 ] * len(pvalues)
            pvalues = pvalues[:,0].tolist()
            scores = [float(s[0]) for s in scores]

        return wm_tokens,pvalues[0] 

    def geometry_detect(self, sequence: List[int], start : int, end : int,payload_max:int) -> float:
        """Detect the watermark in a sequence of tokens and return the z value."""
        detect_res, scores = self.voting_score(self.geometry_cover(sequence,payload_max),sequence)
        detect_res = np.array(detect_res)
        sequence = np.array(sequence)

        # voting_score > threshold
        wm_res = np.where(detect_res >= self.Vscore,1,0)
        wm_tokens = sequence[wm_res == 1]

        # IOU_score
        wm_detect = np.where(detect_res >= self.Vscore)[0]
        ground_truth = np.arange(start, end+1)

        if wm_detect.shape[0] <20:
            return wm_tokens,0,None 
    
        detect_set = set(wm_detect)
        ground_truth_set = set(ground_truth)

        unique_elements = detect_set.union(ground_truth_set)
        intersection = np.intersect1d(wm_detect, ground_truth)

        IOU_score = len(intersection) / len(unique_elements)

        scores_no_aggreg = self.get_scores_by_t([wm_tokens], scoring_method="none", payload_max=payload_max)
        scores = self.aggregate_scores(scores_no_aggreg) # p 1
        pvalues = self.get_pvalues(scores_no_aggreg)
        if payload_max:
            # decode payload and adjust pvalues
            payloads = np.argmin(pvalues, axis=1).tolist()
            pvalues = pvalues[:,payloads][0].tolist()
            scores = [float(s[payload]) for s,payload in zip(scores,payloads)]
            M = payload_max+1
            pvalues = [(1 - (1 - pvalue)**M) if pvalue > min(1 / M, 1e-5) else M * pvalue for pvalue in pvalues]
        else:
            payloads = [ 0 ] * len(pvalues)
            pvalues = pvalues[:,0].tolist()
            scores = [float(s[0]) for s in scores]
        

        return wm_tokens, IOU_score, pvalues[0]

class MarylandAligator(MarylandDetectorZ):

    def __init__(self,threshold,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.prev_pred = 0
        self.threshold = threshold

    def detect(self, text : str, start : int, end : int, payload_max : int):
        # y is the red green ground truth
        scores_no_aggreg = self.get_scores_by_text([text], scoring_method="none", payload_max=payload_max)
        n = len(scores_no_aggreg[0])
        scores = scores_no_aggreg[0]
        y = [score[0] for score in scores]
        step = int(n / 30)
        res = []
        # bidirectional circular detection
        for i in range(0, n, step):
            alig1 = aligator.run_aligator(n,y,np.arange(0,n),0,1,pow(10,-5))
            alig2 = aligator.run_aligator(n,y,np.flip(np.arange(0,n)),0,1,pow(10,-5))
            alig = np.nanmean(np.array([alig1,alig2]),axis = 0)
            alig = np.concatenate((alig[n-i:],alig[0:n-i]))
            res.append(alig)
            y = np.concatenate((y[step:],y[0:step]))
        
        alig = np.nanmean(np.array(res), axis=0)
        detect_res = np.where(alig > self.threshold)[0]
        
        if start == None:
            return alig , 1-(len(detect_res)/len(y)),detect_res
        
        ground_truth = np.arange(start, end+1)

        detect_set = set(detect_res)
        ground_truth_set = set(ground_truth)

        unique_elements = detect_set.union(ground_truth_set)
        num_unique_elements = len(unique_elements)

        intersection = np.intersect1d(detect_res, ground_truth)
        IOU_score = len(intersection) / num_unique_elements

        return alig , IOU_score , detect_res

class OpenaiDetector(WmDetector):
    '''
     Adapted from https://github.com/facebookresearch/three_bricks
    '''
    def __init__(self, 
            tokenizer: LlamaTokenizer, 
            ngram: int = 1,
            seed: int = 0,
            seeding: str = 'hash',
            salt_key: int = 35317, 
            **kwargs):
        super().__init__(tokenizer, ngram, seed, seeding, salt_key, **kwargs)
    
    def score_tok(self, ngram_tokens, token_id):
        """ 
        score_t = -log(1 - rt[token_id]])
        The last line shifts the scores by token_id. 
        ex: scores[0] = r_t[token_id]
            scores[1] = (r_t shifted of 1)[token_id]
            ...
        The score for each payload will be given by scores[payload]
        """
        seed = self.get_seed_rng(ngram_tokens)
        self.rng.manual_seed(seed)
        rs = torch.rand(self.vocab_size, generator=self.rng) # n
        scores = -(1 - rs).log().roll(-token_id)
        return scores
 
    def get_pvalue(self, score: float, ntoks: int, eps: float):
        """ from cdf of a gamma distribution """
        pvalue = special.gammaincc(ntoks, score)
        return max(pvalue, eps)

class OpenaiDetectorZ(WmDetector):
    '''
     Adapted from https://github.com/facebookresearch/three_bricks
    '''
    def __init__(self, 
            tokenizer: LlamaTokenizer, 
            ngram: int = 1,
            seed: int = 0,
            seeding: str = 'hash',
            salt_key: int = 35317, 
            **kwargs):
        super().__init__(tokenizer, ngram, seed, seeding, salt_key, **kwargs)
    
    def score_tok(self, ngram_tokens, token_id):
        """ same as OpenaiDetector but using zscore """
        seed = self.get_seed_rng(ngram_tokens)
        self.rng.manual_seed(seed)
        rs = torch.rand(self.vocab_size, generator=self.rng) # n
        scores = -(1 - rs).log().roll(-token_id)
        return scores
 
    def get_pvalue(self, score: float, ntoks: int, eps: float):
        """ from cdf of a normal distribution """
        mu0 = 1
        sigma0 = np.pi / np.sqrt(6)
        zscore = (score/ntoks - mu0) / (sigma0 / np.sqrt(ntoks))
        pvalue = 0.5 * special.erfc(zscore / np.sqrt(2))

        return max(pvalue, eps)

class OpenaiGeometryWmDetector(OpenaiDetectorZ):
    """
    Class for detecting watermarks in long text.

    Args:
        length: Geometry cover starts from this length
        Vscore: Voting_score_threshold. Normarlly set to 0, meaning once detected in one segment, we recognize it as watermarked.
        pvalue: The P_value threshold for one segment.  
    """
    def __init__(self,length,Vscore,pvalue,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.StartLength = length
        self.Vscore = Vscore
        self.pthres = pvalue # pvalue threshold

    def geometry_cover(self,sequence:List[float],payload_max)->List[List[float]]:
            length = len(sequence)
            geometry_scores = []
            i = self.StartLength
            scores_no_aggreg = self.get_scores_by_t([sequence], scoring_method="none", payload_max=payload_max)
            while i<length - self.ngram:
                i *= 2
                i_score = []
                for j in range(0,length - self.ngram,i):
                    total_len = length-j if j+i > length else i
                    if total_len<64 :
                        break
                   
                    pvalues = self.get_pvalues([scores_no_aggreg[0][j:j+total_len]])
                    if payload_max:
                        # decode payload and adjust pvalues
                        payloads = np.argmin(pvalues, axis=1).tolist()
                        pvalues = pvalues[:,payloads][0].tolist() # in fact pvalue is of size 1, but the format could be adapted to take multiple text at the same time
                        # adjust pvalue to take into account the number of tests (2**payload_max)
                        # use exact formula for high values and (more stable) upper bound for lower values
                        M = payload_max+1
                        pvalues = [(1 - (1 - pvalue)**M) if pvalue > min(1 / M, 1e-5) else M * pvalue for pvalue in pvalues]
                    else:
                        payloads = [ 0 ] * len(pvalues)
                        pvalues = pvalues[:,0].tolist()
                    i_score.append(pvalues[0])
                geometry_scores.append(i_score)
            return geometry_scores
    
    def voting_score(self,scores:List[List[float]],sequence:List[int])->List[float]:
        length = len(sequence)
        voting_res =[0.0] * length
        voting_res = np.array(voting_res)
        scores_idx = 0
        
        i = self.StartLength
        
        while i<length - self.ngram:
           
            i *= 2
            tmp_index = len(scores[scores_idx])
            tokens_idx = 0
            for j in range(0,tmp_index):
                
                if scores[scores_idx][j]<self.pthres: 
                    if tokens_idx+i<=length:
                        voting_res[tokens_idx:tokens_idx+i] += 1 
                    else:
                        voting_res[tokens_idx:] += 1
                tokens_idx += i
            scores_idx += 1 
            self.voting_res = voting_res.tolist()
        return voting_res.tolist() ,scores 
    
    def geometry_detect_without_IOU(self, sequence: List[int],payload_max:int) -> float:
        """Detect the watermark in a sequence of tokens and return the pvalue.
           Do not calculate IOU.
        """
        detect_res,scores = self.voting_score(self.geometry_cover(sequence,payload_max),sequence)
        detect_res = np.array(detect_res)
        sequence = np.array(sequence)

        # voting_score > threshold
        # means at least some block > 32 can detect it
        wm_res = np.where(detect_res >= self.Vscore,1,0)
        wm_tokens = sequence[wm_res == 1]

        wm_detect = np.where(detect_res >= self.Vscore)[0]
        if wm_detect.shape[0] < 32:
            return wm_tokens,None

        detect_set = set(wm_detect)

        scores_no_aggreg = self.get_scores_by_t([wm_tokens], scoring_method="none", payload_max=payload_max)
        scores = self.aggregate_scores(scores_no_aggreg,'sum') # p 1
        pvalues = self.get_pvalues(scores_no_aggreg)
        if payload_max:
            # decode payload and adjust pvalues
            payloads = np.argmin(pvalues, axis=1).tolist()
            pvalues = pvalues[:,payloads][0].tolist() # in fact pvalue is of size 1, but the format could be adapted to take multiple text at the same time
            scores = [float(s[payload]) for s,payload in zip(scores,payloads)]
            # adjust pvalue to take into account the number of tests (2**payload_max)
            # use exact formula for high values and (more stable) upper bound for lower values
            M = payload_max+1
            pvalues = [(1 - (1 - pvalue)**M) if pvalue > min(1 / M, 1e-5) else M * pvalue for pvalue in pvalues]
        else:
            payloads = [ 0 ] * len(pvalues)
            pvalues = pvalues[:,0].tolist()
            scores = [float(s[0]) for s in scores]

        return wm_tokens,pvalues[0]

    def geometry_detect(self, sequence: List[int], start : int, end : int,payload_max:int) -> float:
        """Detect the watermark in a sequence of tokens and return the z value."""
        detect_res,scores = self.voting_score(self.geometry_cover(sequence,payload_max),sequence)
        detect_res = np.array(detect_res)
        sequence = np.array(sequence)

        # voting_score > threshold
        # means at least some block > 32 can detect it
        wm_res = np.where(detect_res >= self.Vscore,1,0)
        wm_tokens = sequence[wm_res == 1]
        wm_detect = np.where(detect_res >= self.Vscore)[0]
        if wm_detect.shape[0] < 32:
            return wm_tokens,0,None

        # IOU_score        
        ground_truth = np.arange(start, end+1)

        detect_set = set(wm_detect)
        ground_truth_set = set(ground_truth)

        unique_elements = detect_set.union(ground_truth_set)
        intersection = np.intersect1d(wm_detect, ground_truth)

        IOU_score = len(intersection) / len(unique_elements)

        scores_no_aggreg = self.get_scores_by_t([wm_tokens], scoring_method="none", payload_max=payload_max)
        scores = self.aggregate_scores(scores_no_aggreg,'sum') # p 1
        pvalues = self.get_pvalues(scores_no_aggreg)
        if payload_max:
            # decode payload and adjust pvalues
            payloads = np.argmin(pvalues, axis=1).tolist()
            pvalues = pvalues[:,payloads][0].tolist() # in fact pvalue is of size 1, but the format could be adapted to take multiple text at the same time
            scores = [float(s[payload]) for s,payload in zip(scores,payloads)]
            # adjust pvalue to take into account the number of tests (2**payload_max)
            # use exact formula for high values and (more stable) upper bound for lower values
            M = payload_max+1
            pvalues = [(1 - (1 - pvalue)**M) if pvalue > min(1 / M, 1e-5) else M * pvalue for pvalue in pvalues]
        else:
            payloads = [ 0 ] * len(pvalues)
            pvalues = pvalues[:,0].tolist()
            scores = [float(s[0]) for s in scores]
        return wm_tokens, IOU_score,pvalues[0]

class OpenaiAligator(OpenaiDetectorZ):

    def __init__(self, threshold, *args, **kwargs):
        super().__init__(*args,**kwargs)
        self.prev_pred = 0
        self.threshold = threshold
    def detect(self, text : str, start : int, end : int, payload_max : int):

        scores_no_aggreg = self.get_scores_by_text([text], scoring_method="none", payload_max=payload_max)
        n = len(scores_no_aggreg[0])
        scores = scores_no_aggreg[0]
        y = [score[0] for score in scores]
        step = int(n / 30)
        res = []
        # bidirectional circular detection
        for i in range(0, n, step):
            alig1 = aligator.run_aligator(n,y,np.arange(0,n),0,1,pow(10,-5))
            alig2 = aligator.run_aligator(n,y,np.flip(np.arange(0,n)),0,1,pow(10,-5))
            alig = np.nanmean(np.array([alig1,alig2]),axis = 0)
            alig = np.concatenate((alig[n-i:],alig[0:n-i]))
            res.append(alig)
            y = np.concatenate((y[step:],y[0:step]))
        
        alig = np.nanmean(np.array(res), axis=0)
        detect_res = np.where(alig > self.threshold)[0]
        if start == None:
            return alig , 1-(len(detect_res)/len(y)),detect_res
        
        ground_truth = np.arange(start, end+1)

        detect_set = set(detect_res)
        ground_truth_set = set(ground_truth)

        unique_elements = detect_set.union(ground_truth_set)
        num_unique_elements = len(unique_elements)

        intersection = np.intersect1d(detect_res, ground_truth)
        IOU_score = len(intersection) / num_unique_elements
        return alig , IOU_score , detect_res
    

    def ref_detect(self, text : str, start : int, end : int, payload_max : int):
        scores_no_aggreg = self.get_scores_by_text([text], scoring_method="none", payload_max=payload_max)
        n = len(scores_no_aggreg[0])
        scores = scores_no_aggreg[0]
        y = [score[0] for score in scores]
        
        res = set()
        # bidirectional circular detection
        for length_i in range(50, n):
            for head in range(0,n-length_i):
                scores_i = scores[head:head+length_i]
                pvalues = self.get_pvalues([scores_i])
                if pvalues.shape[1] ==0:
                    pvalues = [1]
                    scores_i = [0]
                elif payload_max:
                    # decode payload and adjust pvalues
                    payloads = np.argmin(pvalues, axis=1).tolist()
                    pvalues = pvalues[:,payloads][0].tolist() # in fact pvalue is of size 1, but the format could be adapted to take multiple text at the same time
                    scores_i = [float(s[payload]) for s,payload in zip(scores_i,payloads)]
                    # adjust pvalue to take into account the number of tests (2**payload_max)
                    # use exact formula for high values and (more stable) upper bound for lower values
                    M = payload_max+1
                    pvalues = [(1 - (1 - pvalue)**M) if pvalue > min(1 / M, 1e-5) else M * pvalue for pvalue in pvalues]
                else:
                    payloads = [ 0 ] * len(pvalues)
                    pvalues = pvalues[:,0].tolist()
                    scores_i = [float(s[0]) for s in scores_i]
                if pvalues[0] < 0.001:
                    detect_res = set(scores_i)
                    res.union(detect_res)
        ground_truth = np.arange(start, end+1)
        ground_truth_set = set(ground_truth)

        unique_elements = res.union(ground_truth_set)
        num_unique_elements = len(unique_elements)

        intersection = res & ground_truth_set
        IOU_score = len(intersection) / num_unique_elements
        detect_ = np.array(sorted(res))    
        return IOU_score , detect_

class GPTWatermarkBase:
    """
    Adapted from https://github.com/XuandongZhao/Unigram-Watermark

    Base class for watermarking distributions with fixed-group green-listed tokens.

    Args:
        fraction: The fraction of the distribution to be green-listed.
        strength: The strength of the green-listing. Higher values result in higher logit scores for green-listed tokens.
        vocab_size: The size of the vocabulary.
        watermark_key: The random seed for the green-listing.
    """

    def __init__(self, fraction: float = 0.5, strength: float = 2.0, vocab_size: int = 50257, watermark_key: int = 0):
        rng = np.random.default_rng(self._hash_fn(watermark_key))
        mask = np.array([True] * int(fraction * vocab_size) + [False] * (vocab_size - int(fraction * vocab_size)))
        rng.shuffle(mask)
        self.green_list_mask = torch.tensor(mask, dtype=torch.float32)
        self.strength = strength
        self.fraction = fraction
        self.vocab_size = vocab_size

    @staticmethod
    def _hash_fn(x: int) -> int:
        """solution from https://stackoverflow.com/questions/67219691/python-hash-function-that-returns-32-or-64-bits"""
        x = np.int64(x)
        return int.from_bytes(hashlib.sha256(x).digest()[:4], 'little')

class GPTWatermarkDetector(GPTWatermarkBase):
    """
    Class for detecting watermarks in a sequence of tokens.

    Args:
        fraction: The fraction of the distribution to be green-listed.
        strength: The strength of the green-listing. Higher values result in higher logit scores for green-listed tokens.
        vocab_size: The size of the vocabulary.
        watermark_key: The random seed for the green-listing.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def _z_score(num_green: int, total: int, fraction: float) -> float:
        """Calculate and return the z-score of the number of green tokens in a sequence."""
        
        return (num_green - fraction * total) / np.sqrt(fraction * (1 - fraction) * total)
    
    @staticmethod
    def _compute_tau(m: int, N: int, alpha: float) -> float:
        """
        Compute the threshold tau for the dynamic thresholding.

        Args:
            m: The number of unique tokens in the sequence.
            N: Vocabulary size.
            alpha: The false positive rate to control.
        Returns:
            The threshold tau.
        """
        factor = np.sqrt(1 - (m - 1) / (N - 1))
        tau = factor * norm.ppf(1 - alpha)
        return tau

    def detect(self, sequence: List[int]) -> float:
        """Detect the watermark in a sequence of tokens and return the z value."""
        green_tokens = int(sum(self.green_list_mask[i] for i in sequence))

        return self._z_score(green_tokens, len(sequence), self.fraction)

    def unidetect(self, sequence: List[int]) -> float:
        """Detect the watermark in a sequence of tokens and return the z value. Just for unique tokens."""
        sequence = list(set(sequence))
        green_tokens = int(sum(self.green_list_mask[i] for i in sequence))
        return self._z_score(green_tokens, len(sequence), self.fraction)
    
    def dynamic_threshold(self, sequence: List[int], alpha: float) -> (bool, float):
        """Dynamic thresholding for watermark detection. True if the sequence is watermarked, False otherwise."""
        z_score = self.unidetect(sequence)
        tau = self._compute_tau(len(list(set(sequence))), self.vocab_size, alpha)
        return z_score > tau, z_score

class GeometryWmDetector(GPTWatermarkDetector):
    """
    Adapted from https://github.com/XuandongZhao/Unigram-Watermark

    Class for detecting watermarks in long text.

    Args:
        fraction: The fraction of the distribution to be green-listed.
        strength: The strength of the green-listing. Higher values result in higher logit scores for green-listed tokens.
        vocab_size: The size of the vocabulary.
        watermark_key: The random seed for the green-listing.
    """
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
    
    def geometry_cover(self,sequence:List[int], fpr_control:float)->List[List[float]]:
        length = len(sequence)
        scores = []
        i = 16
        while i<length:
            i *= 2
            i_score = []
            for j in range(0,length,i):
                total_len = length-j if j+i > length else i
                judge, wmzscore = self.dynamic_threshold(sequence[j:j+total_len], fpr_control)
                if judge: i_score.append(1)
                else: i_score.append(0)
            scores.append(i_score)

        return scores
    
    def voting_score(self,scores:List[List[float]],sequence:List[int])->List[float]:
        length = len(sequence)
        voting_res =[0.0] * length
        voting_res = np.array(voting_res)
        scores_idx = 0
        i = 16
        while i<length:
            i *= 2
            tmp_index = len(scores[scores_idx])
            tokens_idx = 0
            for j in range(0,tmp_index):
                if scores[scores_idx][j]> 0:
                    voting_res[tokens_idx:tokens_idx+i] += 1 
                    
                tokens_idx += i
            scores_idx += 1 
            self.voting_res = voting_res.tolist()
        return voting_res.tolist() ,scores
    
    def geometry_detect(self, sequence: List[int], threshold:int, start : int, end : int, fpr_control:float) -> float:
        """Detect the watermark in a sequence of tokens and return the z value."""
        detect_res,scores = self.voting_score(self.geometry_cover(sequence,fpr_control),sequence)

        detect_res = np.array(detect_res)
        sequence = np.array(sequence)

        # voting_score > threshold
        # means at least some block > 32 can detect it
        wm_res = np.where(detect_res > threshold,1,0)
        wm_tokens = sequence[wm_res == 1]

        if len(wm_tokens) == 0:
            wm_zscore = 0
        else:
            wm_zscore = self._z_score(sum(self.green_list_mask[i] for i in wm_tokens),len(wm_tokens),self.fraction)

        if start ==None:
            none_wm_tokens = sequence[wm_res == 0]
            IOU_score = len(none_wm_tokens)/len(sequence)
            return wm_tokens, IOU_score, wm_zscore
        # IOU_score
        wm_detect = np.where(detect_res > threshold)[0]
        ground_truth = np.arange(start, end+1)

        detect_set = set(wm_detect)
        ground_truth_set = set(ground_truth)

        unique_elements = detect_set.union(ground_truth_set)
        intersection = np.intersect1d(wm_detect, ground_truth)

        IOU_score = len(intersection) / len(unique_elements)

        return wm_tokens, IOU_score, wm_zscore
    
class Aligator(GPTWatermarkBase): # GPTWatermarkDetector

    def __init__(self, *args, **kwargs): #
        super().__init__(*args, **kwargs)
        self.prev_pred = 0

    def detect(self, sequence : List[int], threshold : float, start : int, end : int):
        # y is the red green ground truth
        y = np.array([self.green_list_mask[i] for i in sequence])
        n = len(y)
        step = int(n / 30)
        res = []
        # bidirectional circular detection
        for i in range(0, n, step):
            alig1 = aligator.run_aligator(n,y,np.arange(0,n),0,1,pow(10,-5))
            alig2 = aligator.run_aligator(n,y,np.flip(np.arange(0,n)),0,1,pow(10,-5))
            alig = np.nanmean(np.array([alig1,alig2]),axis = 0)
            alig = np.concatenate((alig[n-i:],alig[0:n-i]))
            res.append(alig)
            y = np.concatenate((y[step:],y[0:step]))
        
        alig = np.nanmean(np.array(res), axis=0)
        detect_res = np.where(alig > threshold)[0]
        ground_truth = np.arange(start, end+1)

        detect_set = set(detect_res)
        ground_truth_set = set(ground_truth)

        unique_elements = detect_set.union(ground_truth_set)
        num_unique_elements = len(unique_elements)

        intersection = np.intersect1d(detect_res, ground_truth)
        IOU_score = len(intersection) / num_unique_elements
        
        return alig , IOU_score , detect_res
    
    # static detection : fixed threshold
    def static_boundary(self,predict, threshold):
        cumulative_pred = np.cumsum(predict)
        result_set = set()
        i = 32
        while i < len(predict):
            threshold_ = threshold * np.sqrt(self.fraction * (1 - self.fraction) * i) + self.fraction * i
            cumulative_pred_i = cumulative_pred[i-1:-1] - np.concatenate((np.array([0]),cumulative_pred[0:-1-i]))
            pred_i = np.where(cumulative_pred_i > threshold_)[0]
            for x in pred_i:
                indices = set(range(x, x+i))
                result_set = result_set.union(indices)

            i *= 2

        detect_res = np.array(sorted(result_set))
        return detect_res

    def detect2(self,sequence : List[int], threshold : float, start : int, end : int):
        # y is the red green ground truth
        y = np.array([self.green_list_mask[i] for i in sequence])
        n = len(y)
        step = int(n / 30)
        res = []
        # bidirectional circular detection
        for i in range(0, n, step):
            alig1 = aligator.run_aligator(n,y,np.arange(0,n),0,1,pow(10,-5))
            alig2 = aligator.run_aligator(n,y,np.flip(np.arange(0,n)),0,1,pow(10,-5))
            alig = np.nanmean(np.array([alig1,alig2]),axis = 0)
            alig = np.concatenate((alig[n-i:],alig[0:n-i]))
            res.append(alig)
            y = np.concatenate((y[step:],y[0:step]))
        
        # alig_sum = np.zeros_like(alig)
        alig = np.nanmean(np.array(res), axis=0)
        detect_res = self.static_boundary(alig, threshold)
        
        ground_truth = np.arange(start, end+1)

        detect_set = set(detect_res)
        ground_truth_set = set(ground_truth)

        unique_elements = detect_set.union(ground_truth_set)
        num_unique_elements = len(unique_elements)

        intersection = np.intersect1d(detect_res, ground_truth)
        IOU_score = len(intersection) / num_unique_elements
        
        return alig , IOU_score , detect_res
 