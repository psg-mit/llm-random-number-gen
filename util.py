import numpy as np
from dataclasses import dataclass
from typing import Union, List
import functools
import scipy.stats as stats

def build_prefix_tree(items):
    elems = {}
    for (item, item_prob) in items:
        try:
            prefix = item[0]
        except IndexError:
            prefix = None

        try:
            elem = elems[prefix]
        except KeyError:
            elem = []
            elems[prefix] = elem

        elem.append((item[1:], item_prob))

    all_probs = sum(x[1] for x in items)

    tree = {}
    for (k, v) in elems.items():
        suffixes, probs = zip(*v)
        sp = sum(probs) / all_probs
        if k is None:
            tree[k] = (None, sp)
        else:
            tree[k] = (build_prefix_tree(v), sp)

    return tree

@dataclass
class Terminal:
    text: str

@dataclass
class Nonterminal:
    lhs: str
    rhs: List[Union[str, Terminal]]
    prob: float

class Pcfg:
    def __init__(self, nonterminals, start_key, separator):
        self.nonterminals = {}
        self.start_key = start_key
        self.separator = separator

        for nt in nonterminals:
            if nt.lhs not in self.nonterminals:
                self.nonterminals[nt.lhs] = []
            self.nonterminals[nt.lhs].append(nt)

        assert self.start_key in self.nonterminals
        for (k, vs) in self.nonterminals.items():
            assert np.allclose(sum(x.prob for x in vs), 1)
            for v in vs:
                for r in v.rhs:
                    if isinstance(r, str):
                        assert r in self.nonterminals
                    else:
                        assert isinstance(r, Terminal)

    def pretty_print(self):
        res = []
        printed = set()
        to_print = set([self.start_key])
        q = [self.nonterminals[self.start_key]]
        while q:
            nts = q.pop(0)
            printed.add(nts[0].lhs)

            res.append('{} -> {}'.format(
                nts[0].lhs,
                ' | '.join(
                    '{} [{}]'.format(
                        ' '.join(
                            r if isinstance(r, str) else '"{}"'.format(r.text)
                            for r in v.rhs
                        ),
                        v.prob,
                    )
                    for v in nts)))

            for v in nts:
                for r in v.rhs:
                    if isinstance(r, str):
                        nnt = self.nonterminals[r]
                        if nnt[0].lhs not in (printed | to_print):
                            q.append(nnt)
                            to_print.add(nnt[0].lhs)

        return '\n'.join(res)

    @functools.lru_cache()
    def enumerate_with_probability(self, start_key=None):
        if start_key is None:
            start_key = self.start_key

        nts = self.nonterminals[start_key]
        res = []
        for nt in nts:
            strings = []
            for x in nt.rhs:
                if isinstance(x, str):
                    sub_strings = self.enumerate_with_probability(x)
                else:
                    sub_strings = [([x.text], 0.0)]

                if len(strings) == 0:
                    new_strings = sub_strings
                else:
                    new_strings = []
                    for (s1, p1) in strings:
                        for (s2, p2) in sub_strings:
                            new_strings.append((s1 + s2, p1 + p2))

                strings[:] = new_strings
            strings = [(s, p + np.log(nt.prob)) for (s, p) in strings]
            res.extend(strings)
        return res


    def sample(self, n, start_key=None, seed=None, rand=None):
        assert seed is None or rand is None
        if seed is not None:
            rand = np.random.RandomState(seed)
        if rand is None:
            rand = np.random.RandomState()

        if start_key is None:
            start_key = self.start_key

        res = []
        for i in range(n):
            nts = self.nonterminals[start_key]
            prod = rand.choice(nts, p=[x.prob for x in nts])
            m_res = []
            for x in prod.rhs:
                if isinstance(x, str):
                    m_res.extend(self.sample(1, start_key=x, rand=rand))
                else:
                    m_res.append(x.text)
            res.append(self.separator.join(m_res))

        return res


# use normal distribution for numbers
class NormalPcfg(Pcfg):
    def __init__(self, nonterminals, start_key, separator):
        self.nonterminals = {}
        self.start_key = start_key
        self.separator = separator

        for nt in nonterminals:
            if nt.lhs not in self.nonterminals:
                self.nonterminals[nt.lhs] = []
            self.nonterminals[nt.lhs].append(nt)

        assert self.start_key in self.nonterminals
        for (k, vs) in self.nonterminals.items():
           # assert np.allclose(sum(x.prob for x in vs), 1)
            for v in vs:
                for r in v.rhs:
                    if isinstance(r, str):
                        assert r in self.nonterminals
                    else:
                        assert isinstance(r, Terminal)

    def sample(self, n, start_key=None, seed=None, rand=None):
        assert seed is None or rand is None
        if seed is not None:
            rand = np.random.RandomState(seed)
        if rand is None:
            rand = np.random.RandomState()

        elements = np.linspace(0.00, .99, 100)
        sigma = np.std(elements)
        mu = .50
        res = []

        for i in range(n):
            probs = stats.norm.pdf(elements, loc=mu, scale=sigma)
            ch = np.random.choice(elements, p=probs / probs.sum())
            ch = round(ch, 2)
            res.append(str(ch))

        return res

    @functools.lru_cache()
    def enumerate_with_probability(self):
        elements = np.linspace(0.00, .99, 100)
        sigma = np.std(elements)
        mu = .50

        probs = stats.norm.pdf(elements, loc=mu, scale=sigma)
        probs = probs / probs.sum()

        res = []
        for i in range(len(elements)):
            res.append(([str(round(elements[i], 2))], np.log(probs[i])))
        return res

catdog_pcfg = Pcfg([
    Nonterminal('S', ['NP', 'VP'], 1.0),
    Nonterminal('NP', ['Det', 'N'], 0.6),
    Nonterminal('NP', ['N'], 0.4),
    Nonterminal('VP', ['V', 'NP'], 0.8),
    Nonterminal('VP', ['V'], 0.2),
    Nonterminal('Det', [Terminal('the')], 0.7),
    Nonterminal('Det', [Terminal('a')], 0.3),
    Nonterminal('N', [Terminal('cat')], 0.4),
    Nonterminal('N', [Terminal('dog')], 0.3),
    Nonterminal('N', [Terminal('mouse')], 0.2),
    Nonterminal('N', [Terminal('book')], 0.1),
    Nonterminal('V', [Terminal('liked')], 0.5),
    Nonterminal('V', [Terminal('ate')], 0.3),
    Nonterminal('V', [Terminal('read')], 0.2),
], 'S', ' ')

number_pcfg = Pcfg([
    Nonterminal('S', [Terminal('0.'), 'N', 'N'], 1.0),
    Nonterminal('N', [Terminal('0')], 0.1),
    Nonterminal('N', [Terminal('1')], 0.1),
    Nonterminal('N', [Terminal('2')], 0.1),
    Nonterminal('N', [Terminal('3')], 0.1),
    Nonterminal('N', [Terminal('4')], 0.1),
    Nonterminal('N', [Terminal('5')], 0.1),
    Nonterminal('N', [Terminal('6')], 0.1),
    Nonterminal('N', [Terminal('7')], 0.1),
    Nonterminal('N', [Terminal('8')], 0.1),
    Nonterminal('N', [Terminal('9')], 0.1)
], 'S', '')

number_normal_pcfg = NormalPcfg([
    Nonterminal('S', [Terminal('0.'), 'N', 'N'], 1.0),
    Nonterminal('N', [Terminal('0')], 0.0001),
    Nonterminal('N', [Terminal('1')], 0.0015),
    Nonterminal('N', [Terminal('2')], 0.0235),
    Nonterminal('N', [Terminal('3')], 0.135),
    Nonterminal('N', [Terminal('4')], 0.34),
    Nonterminal('N', [Terminal('5')], 0.34),
    Nonterminal('N', [Terminal('6')], 0.135),
    Nonterminal('N', [Terminal('7')], 0.0235),
    Nonterminal('N', [Terminal('8')], 0.0015),
    Nonterminal('N', [Terminal('9')], 0.0001),
], 'S', '')


bits_uniform_pcfg = Pcfg([
    Nonterminal('S', [Terminal('0')], 0.5),
    Nonterminal('S', [Terminal('1')], 0.5),
], 'S', '')

bits_nonuniform_pcfg = Pcfg([
    Nonterminal('S', [Terminal('0')], 0.75),
    Nonterminal('S', [Terminal('1')], 0.25),
], 'S', '')
