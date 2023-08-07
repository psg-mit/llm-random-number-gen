# Can LLMs Generate Random Numbers? Evaluating LLM Sampling in Controlled Domains

### Citation

```
@inproceedings{llm_sampling_renda_hopkins_2023,
title={Can {LLM}s Generate Random Numbers? Evaluating {LLM} Sampling in Controlled Domains},
author={Renda, Alex and Hopkins, Aspen K. and Carbin, Michael},
booktitle={ICML 2023 Workshop: Sampling and Optimization in Discrete Space},
year={2023},
url={http://people.csail.mit.edu/renda/llm-sampling-paper},
}
```


# Contents of this repository

This repository contains the bulk of the code to reproduce the data (`./main.py`) and figures (`./plot.py`) in the paper.
It does not contain any of the models evaluated.

The experiments in the paper used [llama.cpp at commit 7f0e9a77](https://github.com/ggerganov/llama.cpp/tree/7f0e9a775ecc4c6ade271c217f63d6dc93e79eaa) and [llama-cpp-python at commit a1b2d5c0](https://github.com/abetlen/llama-cpp-python/tree/a1b2d5c09b9061e265b504fc6307559f89a8589c). However, we have added (relatively untested) support for Huggingface models as well.

# Setup

## Install dependencies

```bash
pip install -r requirements.txt
```

If you're using `llama.cpp` models, you'll also need to install `llama-cpp-python` and have it available in your Python environment.

## Download models

Download the models you want to evaluate and place them in the `models` directory. The models evaluated in the paper are the 8-bit quantized versions of the `llama.cpp` models, stored in `models/llama-7B/ggml-model-q8_0.bin`, `models/alpaca-7B/ggml-model-q8_0.bin`, `models/llama-13B/ggml-model-q8_0.bin`, etc.

# Running the Code

## Kick-the-tires

The simplest experiment is to run `bert-tiny-uncased` on the uniform bits domain:

```
./main.py --model bert-tiny-uncased --experiment bits_uniform --prompt-examples 0 1 2 3 4 5 6 7 8 9 10 --trials 10 autoregressive --samples 10 --rollouts 10
./main.py --model bert-tiny-uncased --experiment bits_uniform --prompt-examples 0 1 2 3 4 5 6 7 8 9 10 --trials 10 oneshot
./plot.py --domains bits_uniform --show --models bert-tiny-uncased
```

## Generating Data

To evaluate each different sampling technique, run:

`./main.py --model {MODEL_NAME} --experiment {EXPERIMENT NAME} --prompt-examples {LIST OF PROMPT EXAMPLES TO EVALUATE WITH} --trials {NUMBER OF TRIALS TO RUN} (oneshot | autoregressive --rollouts {NUMBER OF ROLLOUTS} --samples {NUMBER OF SAMPLES PER ROLLOUTS} )`

Where:
* `MODEL NAME` is a model name listed [here](https://github.com/psg-mit/llm-random-number-gen/blob/main/main.py#L173-L180)
* `EXPERIMENT NAME` is an experiment listed [here](https://github.com/psg-mit/llm-random-number-gen/blob/main/main.py#L182-L189)
* `PROMPT EXAMPLES` is a list of numbers (e.g., `0`, `0 1`, `0 1 2 3 4 5 6 7 8 9 10`), which specifies how many prompt examples to evaluate with (providing multiple numbers runs multiple distinct experiments)
* `TRIALS` is the number of trials to run per configuration
* `oneshot` says to run NARS experiments
* `autoregressive` says to run ARS experiments
  * `ROLLOUTS` is the number of rollouts to run in a given autoregressive experiment
  * `SAMPLES` is the number of samples to generate per rollout


## Plotting Data

To plot the data, run `./plot.py --domains {DOMAINS} [--models {MODELS}] [--show]`

## Manually Inspecting Data

Data is saved in the `data/` directory.

For one-shot (NARS) data:
```
with open('data/bits_uniform/oneshot/prompt-0/bert-tiny-uncased/trial-0/pcfg-logprobs.pkl', 'rb') as f:
    data = pickle.load(f)
print(data)
```
Results in:
```
([(['0'], -0.6931471805599453), (['1'], -0.6931471805599453)],
 [-7.413161754608154, -6.99363374710083])
```
where the first element of the tuple is the ground-truth log-probabilities for each element of the domain, and the second element is the predicted log-probabilities for those elements.

For autoregressive (ARS) data:
```
with open('data/bits_uniform/oneshot/prompt-0/bert-tiny-uncased/trial-0/rollouts.pkl', 'rb') as f:
    data = pickle.load(f)
print(data)
```
will print a list of lists. Each list represents the samples in a given rollout.

# Editing the Code

You'll likely have to edit the code to do anything other than replicate the entirety of the paper experiments.

## Adding new models

You can add new models by editing [here](https://github.com/psg-mit/llm-random-number-gen/blob/main/main.py#L173-L180).

## Adding new domains

You can add new domains by adding a new entry [here](https://github.com/psg-mit/llm-random-number-gen/blob/main/main.py#L182-L189).
In the current framework, each domain is represented by a PCFG in `util.py`.
