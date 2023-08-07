#!/usr/bin/env python3

from plot_utils import *
import argparse
import builtins
import scipy.stats
import pickle
import pandas as pd
from util import *
import collections
import matplotlib.ticker as mtick
import functools
import os

os.makedirs('figures', exist_ok=True)

all_models = [
    ('Alpaca', '7B'),
    ('LLaMa', '7B'),
    ('LLaMa', '13B'),
    ('LLaMa', '30B'),
    ('LLaMa', '65B'),
]


all_models = [
    ('gpt2', 'tiny'),
]

all_prompt_nums = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

configurations = [
    {
        'name': 'pcfg',
        'format': 'PCFG',
        'ylim': (0.4, 1.25),
        'yticks': [0.5, 0.75, 1],
        'models': all_models,
        'prompt_nums': all_prompt_nums,
        'pcfg': catdog_pcfg,
    },
    {
        'name': 'numbers',
        'format': 'Uniform [0,1)',
        # 'ylim': (0, 2),
        # 'yticks': [0, 0.5, 1, 1.5, 2],
        'ylim': (0, 1.25),
        'yticks': [0, 0.25, 0.5, 0.75, 1],
        'models': all_models,
        'prompt_nums': all_prompt_nums,
        'pcfg': number_pcfg,
    },
    {
        'name': 'numbers_normal',
        'format': 'Normal [0,1)',
        'ylim': (0, 2),
        'yticks': [0, 0.5, 1, 1.5, 2],
        'models': all_models,
        'prompt_nums': all_prompt_nums,
        'pcfg': number_normal_pcfg,
    },
    {
        'name': 'bits_uniform',
        'format': 'Uniform Bits',
        'ylim': (0, 2),
        'yticks': [0, 0.5, 1, 1.5, 2],
        'models': all_models,
        'prompt_nums': all_prompt_nums,
        'pcfg': bits_uniform_pcfg,
    },
    {
        'name': 'bits_nonuniform',
        'format': 'Nonuniform Bits',
        'ylim': (0, 2),
        'yticks': [0, 0.5, 1, 1.5, 2],
        'models': all_models,
        'prompt_nums': all_prompt_nums,
        'pcfg': bits_nonuniform_pcfg,
    },
    {
        'name': 'bits_nonuniform_evil',
        'format': 'Nonuniform Bits (Bad Prompt)',
        'ylim': (0, 2),
        'yticks': [0, 0.5, 1, 1.5, 2],
        'models': all_models,
        'prompt_nums': all_prompt_nums,
        'pcfg': bits_nonuniform_pcfg,
    },
]

def open(filename, mode='r'):
    return builtins.open(filename, mode)


def distance_metric(d1, d2):
    d1 = np.array(d1)
    d2 = np.array(d2)
    return np.sum(np.abs(d1 - d2))
distance_name = 'TVD'

def get_pcfg_prob_map(sentences):
    noun_set = ['cat', 'dog', 'mouse', 'book']
    verb_set = ['liked', 'ate', 'read']
    det_set = ['the', 'a']

    observed_nouns = [n for sentence in sentences for n in sentence.split() if n in noun_set]

    noun_counts = collections.Counter(observed_nouns)
    observed_verbs = [v for sentence in sentences for v in sentence.split() if v in verb_set]
    verb_counts = collections.Counter(observed_verbs)
    observed_dets = [d for sentence in sentences for d in sentence.split() if d in det_set]
    det_counts = collections.Counter(observed_dets)

    noun_freqs = [(n, noun_counts[n] / len(observed_nouns)) for n in noun_set]
    # sort by ordering in noun_set
    noun_freqs = sorted(noun_freqs, key=lambda x: noun_set.index(x[0]))
    verb_freqs = [(v, verb_counts[v] / len(observed_verbs)) for v in verb_set]
    verb_freqs = sorted(verb_freqs, key=lambda x: verb_set.index(x[0]))
    det_freqs = [(d, det_counts[d] / len(observed_dets)) for d in det_set]

    np_det_n_prob = len(observed_dets) / len(observed_nouns)
    vp_v_np_prob = len(observed_nouns) / len(observed_verbs) - 1

    induced_pcfg = Pcfg([
        Nonterminal('S', ['NP', 'VP'], 1.0),
        Nonterminal('NP', ['Det', 'N'], np_det_n_prob),
        Nonterminal('NP', ['N'], 1 - np_det_n_prob),
        Nonterminal('VP', ['V', 'NP'], vp_v_np_prob),
        Nonterminal('VP', ['V'], 1 - vp_v_np_prob),
        Nonterminal('Det', [Terminal('the')], det_freqs[0][1]),
        Nonterminal('Det', [Terminal('a')], det_freqs[1][1]),
        Nonterminal('N', [Terminal('cat')], noun_freqs[0][1]),
        Nonterminal('N', [Terminal('dog')], noun_freqs[1][1]),
        Nonterminal('N', [Terminal('mouse')], noun_freqs[2][1]),
        Nonterminal('N', [Terminal('book')], noun_freqs[3][1]),
        Nonterminal('V', [Terminal('liked')], verb_freqs[0][1]),
        Nonterminal('V', [Terminal('ate')], verb_freqs[1][1]),
        Nonterminal('V', [Terminal('read')], verb_freqs[2][1]),
    ], 'S', ' ')

    return np.array([np.exp(x[1]) for x in induced_pcfg.enumerate_with_probability()])

def get_numbers_prob_map(sentences):
    first_chars = [str(int(float(x) * 10))[-1] for x in sentences]
    first_char_counts = collections.Counter(first_chars)
    return np.array([first_char_counts.get(c, 0) / len(sentences) for c in '0123456789'])

class Plotter:
    def __init__(self, configuration):
        self.dataset_name = configuration['name']
        self.dataset_formatted_name = configuration['format']
        self.models = configuration['models']
        self.ylim = configuration['ylim']
        self.yticks = configuration['yticks']
        self.prompt_nums = configuration['prompt_nums']
        self.pcfg = configuration['pcfg']



        # get the baseline real data
        with open(f'data/{self.dataset_name}/oneshot/prompt-{self.prompt_nums[0]}/{self.models[0][0].lower()}-{self.models[0][1]}/trial-0/pcfg-logprobs.pkl', 'rb') as f:
            (true, _) = pickle.load(f)
            real_logprobs = np.array([x[1] for x in true])
            real_probs = np.exp(real_logprobs)

            if 'numbers' in self.dataset_name:
                xs = self.pcfg.enumerate_with_probability()
                real_probs = [0 for _ in range(10)]
                for x in xs:
                    real_probs[int(self.pcfg.separator.join(x[0])[2])] += np.exp(x[1])
                real_probs = np.array(real_probs)
                # real_probs = np.array([0.1 for _ in range(10)])

        self.real_probs = real_probs

        rs = np.random.RandomState(0)
        uniform_distances = []
        uniform_crosses = []
        for _ in range(10000):
            # sample from simplex
            uniform_sample_1 = np.random.exponential(scale=1.0, size=len(real_probs))
            uniform_sample_1 /= np.sum(uniform_sample_1)
            uniform_sample_2 = np.random.exponential(scale=1.0, size=len(real_probs))
            uniform_sample_2 /= np.sum(uniform_sample_2)

            uniform_distances.append(distance_metric(real_probs, uniform_sample_1))
            uniform_distances.append(distance_metric(real_probs, uniform_sample_2))
            uniform_crosses.append(distance_metric(uniform_sample_1, uniform_sample_2))

        self.uniform_distance = np.mean(uniform_distances)
        self.uniform_cross = np.mean(uniform_crosses)

        self.autoregressive_baseline_distance = self.uniform_distance
        self.autoregressive_baseline_cross = self.uniform_cross


        if self.dataset_name == 'pcfg':
            autoregressive_baseline_distances = []
            autoregressive_baseline_crosses = []
            rs = np.random.RandomState(0)

            for _ in range(1000):
                distrs = []
                for _ in range(2):
                    np_det_n_prob = rs.rand()
                    vp_det_n_prob = rs.rand()
                    det_prob = rs.rand()
                    noun_prob = np.random.exponential(scale=1.0, size=4)
                    noun_prob /= np.sum(noun_prob)
                    verb_prob = np.random.exponential(scale=1.0, size=3)
                    verb_prob /= np.sum(verb_prob)
                    induced_pcfg = Pcfg([
                        Nonterminal('S', ['NP', 'VP'], 1.0),
                        Nonterminal('NP', ['Det', 'N'], np_det_n_prob),
                        Nonterminal('NP', ['N'], 1 - np_det_n_prob),
                        Nonterminal('VP', ['V', 'NP'], vp_det_n_prob),
                        Nonterminal('VP', ['V'], 1 - vp_det_n_prob),
                        Nonterminal('Det', [Terminal('the')], det_prob),
                        Nonterminal('Det', [Terminal('a')], 1 - det_prob),
                        Nonterminal('N', [Terminal('cat')], noun_prob[0]),
                        Nonterminal('N', [Terminal('dog')], noun_prob[1]),
                        Nonterminal('N', [Terminal('mouse')], noun_prob[2]),
                        Nonterminal('N', [Terminal('book')], noun_prob[3]),
                        Nonterminal('V', [Terminal('liked')], verb_prob[0]),
                        Nonterminal('V', [Terminal('ate')], verb_prob[1]),
                        Nonterminal('V', [Terminal('read')], verb_prob[2]),
                    ], 'S', ' ')
                    distr = np.array([np.exp(x[1]) for x in induced_pcfg.enumerate_with_probability()])
                    distrs.append(distr)

                autoregressive_baseline_distances.append(distance_metric(real_probs, distrs[0]))
                autoregressive_baseline_distances.append(distance_metric(real_probs, distrs[1]))
                autoregressive_baseline_crosses.append(distance_metric(distrs[0], distrs[1]))

            self.autoregressive_baseline_distance = np.mean(autoregressive_baseline_distances)
            self.autoregressive_baseline_cross = np.mean(autoregressive_baseline_crosses)

            self.uniform_distance = self.autoregressive_baseline_distance
            self.uniform_cross = self.autoregressive_baseline_cross

    def read_oneshot(self, model_arch, model_size, plot_per_trial):
        y_gt = []
        y_gt_errs = []
        y_cross = []
        y_cross_errs = []
        containment = []
        containment_errs = []

        for pnum in self.prompt_nums:
            all_pred_probs = []
            all_containment = []

            merrs = []

            for trial in range(10):
                with open(f'data/{self.dataset_name}/oneshot/prompt-{pnum}/{model_arch.lower()}-{model_size}/trial-{trial}/pcfg-logprobs.pkl', 'rb') as f:
                    (_, pred_logprobs) = pickle.load(f)
                    pred_logprobs = np.array(pred_logprobs)
                    pred_probs = np.exp(pred_logprobs)

                    if 'numbers' in self.dataset_name:
                        pred_probs = np.array([np.sum(pred_probs[i:i+10]) for i in range(0, len(pred_probs), 10)])

                    all_containment.append(np.sum(pred_probs))

                    if self.dataset_name == 'pcfg':
                        sns = [self.pcfg.separator.join(x[0]) for x in self.pcfg.enumerate_with_probability()]
                        idxs = np.random.choice(np.arange(len(sns)), 10000, p=np.exp(pred_logprobs) / np.exp(pred_logprobs).sum())
                        pred_probs = get_pcfg_prob_map([sns[i] for i in idxs])

                    pred_probs = pred_probs / np.sum(pred_probs)
                    all_pred_probs.append(pred_probs)


                merrs.append((distance_metric(self.real_probs, pred_probs), trial))

                if not plot_per_trial:
                    continue

                m_fig = plt.figure()
                m_ax = m_fig.add_subplot(111)
                plural = 's' if pnum != 1 else ''
                if 'numbers' in self.dataset_name:
                    m_ax.bar(np.linspace(.05, .95, 10), pred_probs, width=0.08)
                    m_ax.set_title(f'{model_arch}-{model_size}, {pnum} Example{plural}, NARS')
                    m_ax.set_xlabel('Generated Number')
                    m_ax.set_ylabel('Sampling Prob.')
                    m_ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

                    if self.dataset_name == 'numbers':
                        m_ax.axhline(y=0.1, color='grey', linestyle='--', alpha=0.5)
                    elif self.dataset_name == 'numbers_normal':
                        for i, x in enumerate(np.linspace(.05, .95, 10)):
                            m_ax.plot([x-0.05,x+0.05], [self.real_probs[i], self.real_probs[i]], color='grey', linestyle='--', alpha=0.5)

                    format_axes(m_ax)

                    m_fig.tight_layout()
                    m_fig.savefig(f'figures/unified-{self.dataset_name}-oneshot-{model_arch}-{model_size}-prompt-{pnum}-trial-{trial}.pdf', bbox_inches='tight')
                    plt.close(m_fig)
                elif 'bits' in self.dataset_name:
                    m_ax.bar([0, 1], pred_probs, width=0.8)
                    m_ax.set_title(f'{model_arch}-{model_size}, {pnum} Example{plural}, NARS')
                    m_ax.set_xlabel('Generated Bit')
                    m_ax.set_ylabel('Sampling Prob.')
                    m_ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

                    if self.dataset_name == 'bits_uniform':
                        m_ax.axhline(y=0.5, color='grey', linestyle='--', alpha=0.5)
                    elif 'bits_nonuniform' in self.dataset_name:
                        plt.plot([0-0.5, 0+0.5], [0.75, 0.75], color='grey', linestyle='--', alpha=0.5)
                        plt.plot([1-0.5, 1+0.5], [0.25, 0.25], color='grey', linestyle='--', alpha=0.5)
                    plt.xticks([0, 1], ['0', '1'])
                    format_axes(m_ax)
                    m_fig.tight_layout()
                    m_fig.savefig(f'figures/unified-{self.dataset_name}-oneshot-{model_arch}-{model_size}-prompt-{pnum}-trial-{trial}.pdf', bbox_inches='tight')
                    plt.close(m_fig)

            gt_distances = np.array([distance_metric(self.real_probs, pred_probs) for pred_probs in all_pred_probs])
            gt_distances /= self.uniform_distance

            if len(gt_distances) > 0 and plot_per_trial and ('numbers' in self.dataset_name or 'bits' in self.dataset_name):
                merrs, mtridxs = zip(*sorted(merrs))
                merrs = np.array(merrs)
                mtridxs = np.array(mtridxs)
                mtridx = mtridxs[len(mtridxs) // 2]
                print(f'{self.dataset_name} {model_arch}-{model_size} prompt {pnum} oneshot median distance idx: {mtridx}')

                shutil.copy2(f'figures/unified-{self.dataset_name}-oneshot-{model_arch}-{model_size}-prompt-{pnum}-trial-{mtridx}.pdf', f'figures/unified-{self.dataset_name}-oneshot-{model_arch}-{model_size}-prompt-{pnum}-trial-median.pdf')

            cross_distances = np.array([distance_metric(all_pred_probs[i], all_pred_probs[j]) for i in range(len(all_pred_probs)) for j in range(i+1, len(all_pred_probs))])
            cross_distances /= self.uniform_cross

            y_gt.append(np.mean(gt_distances))
            y_gt_errs.append(np.std(gt_distances) / np.sqrt(len(gt_distances)))
            y_cross.append(np.mean(cross_distances))
            y_cross_errs.append(np.std(cross_distances) / np.sqrt(len(cross_distances)))
            containment.append(np.mean(all_containment))
            containment_errs.append(np.std(all_containment) / np.sqrt(len(all_containment)))

        return y_gt, y_gt_errs, y_cross, y_cross_errs, containment, containment_errs

    def read_autoregressive(self, model_arch, model_size, plot_per_trial, is_raw):
        y_autoregressive_gt = []
        y_autoregressive_gt_errs = []
        y_autoregressive_cross = []
        y_autoregressive_cross_errs = []
        autoregressive_containment = []
        autoregressive_containment_errs = []

        if is_raw:
            raw_dash = 'raw-'
        else:
            raw_dash = ''

        for pnum in self.prompt_nums:
            all_pred_probs = []
            all_containment = []
            merrs = []

            for trial in range(10):
                try:
                    with open(f'data/{self.dataset_name}/autoregressive/prompt-{pnum}/{model_arch.lower()}-{model_size}/trial-{trial}/{raw_dash}rollouts.pkl', 'rb') as f:
                        rollouts = pickle.load(f)
                except FileNotFoundError:
                    print('FILE NOT FOUND: {}'.format(f'data/{self.dataset_name}/autoregressive/prompt-{pnum}/{model_arch.lower()}-{model_size}/trial-{trial}/{raw_dash}rollouts.pkl'))
                    rollouts = [['bad'] * 10] * 10

                data = [x for rollout in rollouts for x in rollout]
                data = list(map(lambda x: x.strip(' .'), data))

                if 'numbers' in self.dataset_name:
                    legal_data = []
                    for s in data:
                        try:
                            if 0 <= float(s) < 1:
                                legal_data.append(s)
                        except ValueError:
                            continue
                else:
                    legal_sentences = set(self.pcfg.separator.join(x[0]) for x in self.pcfg.enumerate_with_probability())
                    legal_data = [x for x in data if x in legal_sentences]

                all_containment.append(len(legal_data) / len(data))
                if len(legal_data) == 0:
                    continue

                if 'numbers' in self.dataset_name:
                    pred_probs = get_numbers_prob_map(legal_data)
                elif self.dataset_name == 'pcfg':
                    pred_probs = get_pcfg_prob_map(legal_data)
                else:
                    legal_data_counts = collections.Counter(legal_data)
                    sentences = [self.pcfg.separator.join(x[0]) for x in self.pcfg.enumerate_with_probability()]
                    pred_probs = np.array([legal_data_counts[x] if x in legal_data_counts else 0 for x in sentences])
                    pred_probs = pred_probs / np.sum(pred_probs)

                pred_probs = pred_probs / np.sum(pred_probs)
                all_pred_probs.append(pred_probs)

                merrs.append((distance_metric(self.real_probs, pred_probs), trial))

                if not plot_per_trial:
                    continue

                m_fig = plt.figure()
                m_ax = m_fig.add_subplot(111)
                plural = 's' if pnum != 1 else ''
                if 'numbers' in self.dataset_name:
                    m_ax.bar(np.linspace(.05, .95, 10), pred_probs, width=0.08)

                    if self.dataset_name == 'numbers':
                        m_ax.axhline(y=0.1, color='grey', linestyle='--', alpha=0.5)
                    elif self.dataset_name == 'numbers_normal':
                        for i, x in enumerate(np.linspace(.05, .95, 10)):
                            m_ax.plot([x-0.05,x+0.05], [self.real_probs[i], self.real_probs[i]], color='grey', linestyle='--', alpha=0.5)

                    m_ax.set_title(f'{model_arch}-{model_size}, {pnum} Example{plural}, ARS')
                    m_ax.set_xlabel('Generated Number')
                    m_ax.set_ylabel('Sampling Prob.')
                    m_ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
                    format_axes(m_ax)
                    m_fig.tight_layout()
                    m_fig.savefig(f'figures/unified-{self.dataset_name}-autoregressive-{model_arch}-{model_size}-prompt-{pnum}-trial-{trial}.pdf', bbox_inches='tight')
                    plt.close(m_fig)
                elif 'bits' in self.dataset_name:
                    m_ax.bar([0, 1], pred_probs, width=0.8)
                    m_ax.set_title(f'{model_arch}-{model_size}, {pnum} Example{plural}, ARS')
                    m_ax.set_xlabel('Generated Bit')
                    m_ax.set_ylabel('Sampling Prob.')
                    m_ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

                    if self.dataset_name == 'bits_uniform':
                        m_ax.axhline(y=0.5, color='grey', linestyle='--', alpha=0.5)
                    elif 'bits_nonuniform' in self.dataset_name:
                        plt.plot([0-0.5, 0+0.5], [0.75, 0.75], color='grey', linestyle='--', alpha=0.5)
                        plt.plot([1-0.5, 1+0.5], [0.25, 0.25], color='grey', linestyle='--', alpha=0.5)
                    plt.xticks([0, 1], ['0', '1'])
                    format_axes(m_ax)
                    m_fig.tight_layout()
                    m_fig.savefig(f'figures/unified-{self.dataset_name}-autoregressive-{model_arch}-{model_size}-prompt-{pnum}-trial-{trial}.pdf', bbox_inches='tight')
                    plt.close(m_fig)


            gt_distances = np.array([distance_metric(self.real_probs, pred_probs) for pred_probs in all_pred_probs])
            gt_distances /= self.autoregressive_baseline_distance

            if len(gt_distances) > 0 and plot_per_trial and ('numbers' in self.dataset_name or 'bits' in self.dataset_name):
                merrs, mtridxs = zip(*sorted(merrs))
                merrs = np.array(merrs)
                mtridxs = np.array(mtridxs)
                mtridx = mtridxs[len(mtridxs) // 2]
                print(f'{self.dataset_name} {model_arch}-{model_size} prompt {pnum} autoregressive median distance idx: {mtridx}')

                median_fname = f'figures/unified-{self.dataset_name}-autoregressive-{model_arch}-{model_size}-prompt-{pnum}-trial-median.pdf'
                shutil.copy2(f'figures/unified-{self.dataset_name}-autoregressive-{model_arch}-{model_size}-prompt-{pnum}-trial-{mtridx}.pdf', median_fname)

            cross_distances = np.array([distance_metric(all_pred_probs[i], all_pred_probs[j]) for i in range(len(all_pred_probs)) for j in range(i+1, len(all_pred_probs))])
            cross_distances /= self.autoregressive_baseline_cross

            y_autoregressive_gt.append(np.mean(gt_distances))
            y_autoregressive_gt_errs.append(np.std(gt_distances) / np.sqrt(len(gt_distances)))
            y_autoregressive_cross.append(np.mean(cross_distances))
            y_autoregressive_cross_errs.append(np.std(cross_distances) / np.sqrt(len(cross_distances)))
            autoregressive_containment.append(np.mean(all_containment))
            autoregressive_containment_errs.append(np.std(all_containment) / np.sqrt(len(all_containment)))

        return y_autoregressive_gt, y_autoregressive_gt_errs, y_autoregressive_cross, y_autoregressive_cross_errs, autoregressive_containment, autoregressive_containment_errs

    def do_plot(self, n_columns, labels_and_funcs):
        assert len(labels_and_funcs) % n_columns == 0

        # add apadding between columns
        fig = plt.figure(figsize=(9 * n_columns, 13))
        axes = fig.subplots(3, n_columns, squeeze=False, sharex='col', sharey='row')


        LS = ['-', '--', ':']
        MARKERS = ['o', 'x', '^']

        for model_idx, (model_arch, model_size) in enumerate(self.models):
            for lab_i, (label, func) in enumerate(labels_and_funcs):
                dist, dist_errs, cross, cross_errs, containment, containment_errs = func(model_arch, model_size)
                plot_label=f'{model_arch}-{model_size}'
                if n_columns == 1:
                    ls = LS[lab_i]
                    marker = MARKERS[lab_i]
                    plot_label = f'{plot_label} {label}'
                else:
                    ls = LS[0]
                    marker = MARKERS[0]

                for i, (y, errs) in enumerate([
                        (dist, dist_errs),
                        (cross, cross_errs),
                        (containment, containment_errs),
                        ]):
                    ax = axes[i, lab_i % n_columns]

                    legal_idxs = np.logical_and(np.logical_not(np.isnan(y)), np.logical_not(np.isinf(y)))
                    legal_prompt_nums = np.array(self.prompt_nums)[legal_idxs]
                    legal_y = np.array(y)[legal_idxs]
                    legal_y_errs = np.array(errs)[legal_idxs]
                    ax.plot(legal_prompt_nums, legal_y, label=plot_label, marker=marker, linestyle=ls, color=C[model_idx], ms=8)
                    ax.fill_between(legal_prompt_nums, legal_y - legal_y_errs, legal_y + legal_y_errs, alpha=0.2, color=C[model_idx])

        for i in range(n_columns):
            axes[2, i].set_xlabel('Number of Prompt Examples')



        for ax in [axes[0][0], axes[1][0]]:
            ax.set_ylabel('{} Distance \n (Normalized)'.format(distance_name), linespacing=1.2)


        for j in range(n_columns):
            for i in range(3):
                # axes[i][j].set_xlim(0, 10)
                axes[i][j].yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

            for i in [0, 1]:
                axes[i,j].set_ylim(*self.ylim)
                axes[i,j].set_yticks(self.yticks)

            axes[2,j].set_ylim(0, 1)
            axes[2,j].set_yticks([0, 0.25, 0.5, 0.75, 1.0])


        axes[2][0].set_ylabel('Containment')

        for i in range(n_columns):
            if n_columns == 1:
                m_label = self.dataset_formatted_name
            else:
                m_label = f'{self.dataset_formatted_name} {labels_and_funcs[i][0]}'

            axes[0][i].set_title(f'{m_label} Error')
            axes[1][i].set_title(f'{m_label} Variance')
            axes[2][i].set_title(f'{m_label} Containment')

        fig.tight_layout()
        fig.subplots_adjust(wspace=0.1, hspace=0.5)

        for ax in fig.get_axes():
            format_axes(ax)

        labels, _ = zip(*labels_and_funcs)
        all_labels = '-'.join(l.replace(' ', '_') for l in labels)
        fig.savefig(f'figures/{self.dataset_name}-{n_columns}-{all_labels}-error-variance-containment.pdf', bbox_inches='tight')
        handles, labels = axes[0,0].get_legend_handles_labels()

        # if 1 column, we need a verstion with len(labels_and_funcs) rows and a version with len(labels_and_funcs) columns
        # if >1 column, we need a version with 1 row and a version with 1 column

        if n_columns == 1:
            legend_fig = plt.figure(figsize=(2.75 * len(labels) // len(labels_and_funcs), 0.5*len(labels_and_funcs)))
            legend_ax = legend_fig.add_subplot(111)
            legend_ax.axis('off')
            legend_ax.legend(handles, labels, loc='center', ncol=len(labels) // len(labels_and_funcs))
            legend_fig.tight_layout()
            legend_fig.savefig(f'figures/unified-legend-{n_columns}-wide.pdf', bbox_inches='tight')

            # deinterleave the handles and labels
            all_handles = []
            all_labels = []
            for i in range(len(labels_and_funcs)):
                all_handles.append(handles[i::len(labels_and_funcs)])
                all_labels.append(labels[i::len(labels_and_funcs)])
            concat_handles = [handle for handles in all_handles for handle in handles]
            concat_labels = [label for labels in all_labels for label in labels]

            legend_fig = plt.figure(figsize=(2.5 * len(labels_and_funcs), 0.45*len(labels) // len(labels_and_funcs)))
            legend_ax = legend_fig.add_subplot(111)
            legend_ax.axis('off')
            legend_ax.legend(concat_handles, concat_labels, loc='center', ncol=len(labels_and_funcs))
            legend_fig.tight_layout()
            legend_fig.savefig(f'figures/unified-legend-{n_columns}-tall.pdf', bbox_inches='tight')

        else:
            legend_fig = plt.figure(figsize=(2 * len(labels), 0.5))
            legend_ax = legend_fig.add_subplot(111)
            legend_ax.axis('off')
            legend_ax.legend(handles, labels, loc='center', ncol=len(labels))
            legend_fig.tight_layout()
            legend_fig.savefig(f'figures/unified-legend-{n_columns}-wide.pdf', bbox_inches='tight')

            legend_fig = plt.figure(figsize=(2, 0.3 * len(labels)))
            legend_ax = legend_fig.add_subplot(111)
            legend_ax.axis('off')
            legend_ax.legend(handles, labels, loc='center', ncol=1)
            legend_fig.tight_layout()
            legend_fig.savefig(f'figures/unified-legend-{n_columns}-tall.pdf', bbox_inches='tight')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--domains', nargs='+', default=['numbers', 'pcfg', 'numbers_normal', 'bits_uniform', 'bits_nonuniform', 'bits_nonuniform_evil'])
    parser.add_argument('--show', action='store_true', default=False)
    parser.add_argument('--plot-per-trial', action='store_true', default=False)
    parser.add_argument('--n-columns', type=int, default=2)
    parser.add_argument('--types', nargs='+', default=['NARS', 'ARS'])

    args = parser.parse_args()

    for name in args.domains:
        configuration = next(c for c in configurations if c['name'].lower() == name.lower())
        plotter = Plotter(configuration)

        plotters = {
            'NARS': functools.partial(plotter.read_oneshot, plot_per_trial=args.plot_per_trial),
            'ARS RAW': functools.partial(plotter.read_autoregressive, plot_per_trial=args.plot_per_trial, is_raw=True),
            'ARS': functools.partial(plotter.read_autoregressive, plot_per_trial=args.plot_per_trial, is_raw=False),
        }

        plotter.do_plot(args.n_columns, [(t, plotters[t]) for t in args.types])

        if args.show:
            plt.show()


if __name__ == '__main__':
    main()
