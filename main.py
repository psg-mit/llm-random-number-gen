#!/usr/bin/env python3

import abc
import argparse
from dataclasses import dataclass
import itertools
import numpy as np
import os
import pickle
import sys
import tqdm.auto as tqdm
from typing import List, Tuple, Dict, Any, Optional, Union
import util

@dataclass
class Result:
    text: str
    tokens: List[str]
    logprobs: List[Dict[str, float]]
    response_start: int

class Runner(abc.ABC):
    def __init__(self, max_gen_tokens_per_line=None, debug=False, **load_kwargs):
        self.max_gen_tokens_per_line = max_gen_tokens_per_line
        self.debug = debug
        self.load_model(**load_kwargs)

    @abc.abstractmethod
    def load_model(self):
        ''' Load the model. '''
        pass

    @abc.abstractmethod
    def check_log_perplexity(self, prefix, suffix):
        ''' Gets the log perplexity of the suffix given the prefix. '''
        pass

    @abc.abstractmethod
    def generate_line(self, prompt, **sampling_params):
        ''' Generates a single line of text. '''
        pass

    def one_shot_experiment(self, pcfg, prompt):
        logprobs = pcfg.enumerate_with_probability()
        pred_logprobs = []

        for (sentence, logprob) in tqdm.tqdm(logprobs, leave=False):
            log_perp = self.check_log_perplexity(prompt, ' ' + pcfg.separator.join(sentence))
            pred_logprobs.append(log_perp)

        return (logprobs, pred_logprobs)

    def autoregressive_experiment(self, prompt, n_samples, **sampling_params):
        numbering = int(prompt.split('\n')[-1][:-1])
        samples = []

        while len(samples) < n_samples:
            text = self.generate_line(prompt, **sampling_params)
            gen_line = text.split('\n')[0]
            samples.append(gen_line)
            numbering += 1
            prompt += '{}\n{}.'.format(gen_line, numbering)

        return samples

class HuggingfaceRunner(Runner):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def load_model(self, tokenizer_name, model_name):
        import transformers, torch
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_name)
        self.llm = transformers.AutoModelForCausalLM.from_pretrained(model_name)
        self.llm.eval()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.llm.to(self.device)

    def check_log_perplexity(self, prefix, suffix):
        # Gets the log perplexity of the suffix given the prefix.
        import torch
        llm = self.llm
        tokenizer = self.tokenizer
        prefix_tokens = tokenizer.encode(prefix, add_special_tokens=False)
        suffix_tokens = tokenizer.encode(suffix, add_special_tokens=False)
        tokens = prefix_tokens + suffix_tokens
        tokens = torch.tensor([tokens])
        tokens = tokens.to(self.device)
        with torch.no_grad():
            logprobs = llm(tokens)[0]
        logprobs = logprobs[0, len(prefix_tokens):len(prefix_tokens) + len(suffix_tokens)]
        suffix_token_logprobs = logprobs[np.arange(len(suffix_tokens)), suffix_tokens]
        return suffix_token_logprobs.sum().item()

    def generate_line(self, prompt, **sampling_params):
        import transformers
        if self.max_gen_tokens_per_line:
            max_new_tokens = self.max_gen_tokens_per_line
        else:
            max_new_tokens = 512 - len(prompt.split('\n')[-1])
        generation_config = transformers.GenerationConfig(**sampling_params, max_new_tokens=max_new_tokens)
        inputs = self.tokenizer.encode(prompt, add_special_tokens=False, return_tensors='pt')
        inputs = inputs.to(self.device)
        outputs = self.llm.generate(inputs, generation_config=generation_config, pad_token_id=self.tokenizer.eos_token_id)
        text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        text = text[len(prompt):]
        return text.split('\n')[0]

class LlamaRunner(Runner):
    def __init__(self, **load_kwargs):
        super().__init__(**load_kwargs)

    def load_model(self, model_name, n_gpu_layers):
        from llama_cpp import Llama, LlamaCache
        cache = LlamaCache()
        self.llm = Llama(model_path=model_name, n_gpu_layers=n_gpu_layers, verbose=False, logits_all=True, n_threads=0)

    def check_log_perplexity(self, prefix, suffix):
        llm = self.llm

        suffix_begin = len(llm.tokenize((' ' + prefix).encode('utf-8')))
        output = llm(prefix + suffix, max_tokens=0, echo=True)

        log_perplexity = 0

        tokens = itertools.chain(itertools.islice(llm.eval_tokens, suffix_begin, None), llm.tokenize('\n'.encode('utf-8'), add_bos=False))
        logits = itertools.islice(llm.eval_logits, suffix_begin - 1, None)
        for token, logit in zip(tokens, logits):
            log_probs = logit[token] - np.log(np.sum(np.exp(logit)))
            if self.debug:
                print('token is "{}", predicted argmax token is "{}"'.format(
                    llm.detokenize([token]),
                    llm.detokenize([np.argmax(logit)])
                ))
            log_perplexity += log_probs

        return log_perplexity


    def generate_line(self, prompt, n_samples, raw=False):
        newline_tok = self.llm.tokenize(b'\n')[1]

        sampling_params = {
            'temperature': 0.8,
            'top_p': 0.95,
            'frequency_penalty': 0.0,
            'presence_penalty': 0.0,
            'repeat_penalty': 1.1,
            'top_k': 40,
            'tfs_z': 1.0,
            'mirostat_mode': 0,
            'mirostat_tau': 5.0,
            'mirostat_eta': 0.1,
        }

        if raw:
            sampling_params['temperature'] = 1.0
            sampling_params['top_p'] = 1.0
            sampling_params['top_k'] = 0
            sampling_params['repeat_penalty'] = 1.0

        def logits_processor(tokens, logits):
            logits[self.llm._token_eos] = -np.inf
            return logits

        stopping_criteria =  lambda toks, _: toks.count(newline_tok) >= 1 + prompt.count('\n')
        prompt_size = len(self.llm.tokenize(prompt.encode('utf-8')))
        max_tokens = 512 - prompt_size
        if self.max_gen_tokens_per_line:
            max_tokens = self.max_gen_tokens_per_line

        return self.llm(prompt, max_tokens=max_tokens, echo=False, stopping_criteria=stopping_criteria, logits_processor=logits_processor, **sampling_params)['choices'][0]['text']

model_params = {
    'llama-7B': ('llama', 'models/llama-7B/ggml-model-q8_0.bin', 1000),
    'alpaca-7B': ('llama', 'models/alpaca-7B/ggml-model-q8_0.bin', 1000),
    'llama-13B': ('llama', 'models/llama-13B/ggml-model-q8_0.bin', 1000),
    'llama-30B': ('llama', 'models/llama-30B/ggml-model-q8_0.bin', 1000),
    'llama-65B': ('llama', 'models/llama-65B/ggml-model-q8_0.bin', 1000),
    'llama-7B-huggingface': ('huggingface', 'decapoda-research/llama-7b-hf', 0),
    'bert-tiny-uncased': ('huggingface', 'google/bert_uncased_L-2_H-128_A-2', 0),
}

experiments = {
    'pcfg': util.catdog_pcfg,
    'numbers': util.number_pcfg,
    'numbers_normal': util.number_normal_pcfg,
    'bits_uniform': util.bits_uniform_pcfg,
    'bits_nonuniform': util.bits_nonuniform_pcfg,
    'bits_nonuniform_evil': util.bits_nonuniform_pcfg,
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=model_params.keys(), required=True)
    parser.add_argument('--experiment', choices=experiments.keys(), required=True)
    parser.add_argument('--prompt-examples', type=int, default=0, nargs='+')
    parser.add_argument('--trials', type=int, default=10)
    parser.add_argument('--debug', action='store_true')

    subparsers = parser.add_subparsers(dest='command', required=True)
    one_shot_parser = subparsers.add_parser('oneshot')
    autoregressive_parser = subparsers.add_parser('autoregressive')

    autoregressive_parser.add_argument('--samples', type=int, required=True)
    autoregressive_parser.add_argument('--rollouts', type=int, default=1)
    autoregressive_parser.add_argument('--raw', action='store_true')

    args = parser.parse_args()

    if 'bits' in args.experiment:
        max_gen_tokens_per_line = 5
    else:
        max_gen_tokens_per_line = 10

    runner_type, model, n_gpu_layers = model_params[args.model]
    pcfg = experiments[args.experiment]
    if runner_type == 'llama':
        runner = LlamaRunner(model_name=model, n_gpu_layers=n_gpu_layers, max_gen_tokens_per_line=max_gen_tokens_per_line, debug=args.debug)
    elif runner_type == 'huggingface':
        runner = HuggingfaceRunner(model_name=model, tokenizer_name='gpt2', max_gen_tokens_per_line=max_gen_tokens_per_line, debug=args.debug)
    else:
        raise ValueError('unknown runner type {}'.format(runner_type))

    for prompt_examples in tqdm.tqdm(args.prompt_examples):
        for trial in tqdm.trange(args.trials, leave=False):
            data_dir = os.path.join('data', args.experiment, args.command, 'prompt-{}'.format(prompt_examples), args.model.lower(), 'trial-{}'.format(trial))
            os.makedirs(data_dir, exist_ok=True)

            if args.experiment == 'pcfg':
                prompt = 'The following is a list of samples from the following PCFG (note that they are not necessarily grammatical English):\n\n```\n{}\n```\n\n'.format(pcfg.pretty_print())
            elif args.experiment == 'numbers':
                prompt = 'The following is a list of uniform random numbers in the interval [0, 1]:\n\n'
            elif args.experiment == 'numbers_normal':
                prompt = 'The following is a list of normally distributed random numbers in the interval [0, 1] with mean 0.5 and std 0.2887:\n\n'
            elif args.experiment == 'bits_uniform':
                prompt = 'The following is a list of uniform random bits (0, 1):\n\n'
            elif args.experiment == 'bits_nonuniform':
                prompt = 'The following is a list of bits of which 75% are 0 and 25% are 1:\n\n'
            elif args.experiment == 'bits_nonuniform_evil':
                prompt = 'The following is a list of bits of which 25% are 0 and 75% are 1:\n\n'
            else:
                raise ValueError('unknown experiment "{}"'.format(args.experiment))

            synthetic_prompt_samples = pcfg.sample(prompt_examples, seed=trial)
            for i, sample in enumerate(synthetic_prompt_samples):
                prompt += '{}. {}\n'.format(i + 1, sample)
            prompt += '{}.'.format(len(synthetic_prompt_samples) + 1)


            if args.debug:
                print('Prompt: {}'.format(prompt))

            if args.command == 'oneshot':
                (logprobs, pred_logprobs) = runner.one_shot_experiment(pcfg, prompt)
                local_fname = os.path.join(data_dir, 'pcfg-logprobs.pkl')

                with open(local_fname, 'wb') as f:
                    pickle.dump((logprobs, pred_logprobs), f)

            elif args.command == 'autoregressive':
                rollouts = []
                for i in tqdm.trange(args.rollouts):
                    rollouts.append(runner.autoregressive_experiment(prompt, args.samples, raw=args.raw))

                bname = ('raw-' if args.raw else'') + 'rollouts.pkl'
                local_fname = os.path.join(data_dir, bname)
                with open(local_fname, 'wb') as f:
                    pickle.dump(rollouts, f)

if __name__ == '__main__':
    main()
