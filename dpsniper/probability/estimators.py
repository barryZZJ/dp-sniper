from dpsniper.mechanisms.abstract import Mechanism
from dpsniper.attack.attack import Attack
from dpsniper.utils.my_logging import log
from dpsniper.utils.my_multiprocessing import the_parallel_executor, split_by_batch_size, split_into_parts
from dpsniper.probability.binomial_cdf import lcb, ucb
from dpsniper.search.ddconfig import DDConfig

import numpy as np
import math
from bitstring import Bits

class MyBits:
    def __init__(self, vals):
        self.vals = Bits(reversed(vals))
    def __hash__(self):
        hash(self.vals.uint)
    def __lt__(self, other):
        return self.vals.uint < other.vals.uint
    def __eq__(self, other):
        return self.vals.uint == other.vals.uint
    def __iter__(self):
        return reversed(self.vals)
    def __repr__(self):
        return str(list(self))

class MyReals:
    def __init__(self, vals):
        if isinstance(vals, float) or isinstance(vals, np.float):
            self.vals = [vals]
        else:
            self.vals = vals
        self.thresh = 1
    def __lt__(self, other):
        sum1 = sum(self.vals)
        sum2 = sum(other.vals)
        return sum1 < sum2
    def __eq__(self, other):
        eq_mask = np.round(self.vals) == np.round(other.vals)
        return eq_mask.all()
    def __iter__(self):
        return self.vals.__iter__()
    def __repr__(self):
        return self.vals.__repr__()
# r = [-195.54639776473698, 1.7573380844718294, 11.98242111053098, 46.35669983374418, 8.498668642844017]
# r2 = [-195.04639776473698, 1.7573380844718294, 11.98242111053098, 46.35669983374418, 8.498668642844017]
# r = MyReals(r)
# r2 = MyReals(r2)
# ar = np.array([r,r2])
# u = np.unique(ar)
# b,b2,b3=MyBits([False,True]), MyBits([True]), MyBits([False,True])
# a = np.array([b,b2,b3])
# r=np.unique(a)

def unique(a, b):
    if isinstance(a, np.ndarray) or isinstance(a, list):
        if isinstance(a[0], MyBits) or isinstance(a[0], MyReals):
            a, indices = np.unique(a, return_index=True)
        else:
            raise NotImplementedError
    else:
        a, indices = np.unique(a, return_index=True, axis=0)

    b = np.take(b, indices)
    return a.tolist(), b.tolist()

class PrEstimator:
    """
    Class for computing an estimate of Pr[M(a) in S].
    """

    def __init__(self, mechanism: Mechanism, n_samples: int, config: DDConfig, use_parallel_executor: bool = False, log_outputs=False):
        """
        Creates an estimator.

        Args:
            mechanism: mechanism
            n_samples: number of samples used to estimate the probability
            use_parallel_executor: whether to use the global parallel executor for probability estimation.
        """
        self.mechanism = mechanism
        self.n_samples = n_samples
        self.use_parallel_executor = use_parallel_executor
        self.config = config
        self.log_outputs = log_outputs

    def compute_pr_estimate(self, a, attack: Attack) -> float:
        """
        Returns:
             An estimate of Pr[M(a) in S]
        """
        log_bs = []
        log_bprobs = []

        if not self.use_parallel_executor:
            res = self._compute_frac_cnt((self, attack, a, self.n_samples))
            frac_cnt, log_bs, log_bprobs = res
        else:
            inputs = [(self, attack, a, batch) for batch in split_into_parts(self.n_samples, self.config.n_processes)]
            res = the_parallel_executor.execute(self._compute_frac_cnt, inputs)
            counts = []
            for count, frac_log_b, frac_log_bprob in res:
                counts.append(count)
                log_bs.extend(frac_log_b)
                log_bprobs.extend(frac_log_bprob)

            frac_cnt = math.fsum(counts)
        if self.log_outputs:
            if len(log_bs) > 0:
                log_bs, log_bprobs = unique(log_bs, log_bprobs)
                if isinstance(log_bs[0], list) or isinstance(log_bs[0], MyBits) or isinstance(log_bs[0], MyReals):
                    log.data('bs', [list(b) for b in log_bs])
                else:
                    log.data('bs', log_bs)
            else:
                log.data('bs', log_bs)
            log.data('bprobs', log_bprobs)

        pr = frac_cnt / self.n_samples
        return pr

    def _get_samples(self, a, n_samples):
        return self.mechanism.m(a, n_samples=n_samples)

    def _check_attack(self, bs, attack):
        return attack.check(bs)

    def _compute_frac_cnt(self, args):
        pr_estimator, attack, a, n_samples = args

        frac_counts = []
        log_bs = []
        log_bprobs = []
        for sequential_size in split_by_batch_size(n_samples, pr_estimator.config.prediction_batch_size):
            bs = pr_estimator._get_samples(a, sequential_size)
            res = pr_estimator._check_attack(bs, attack)

            if self.log_outputs:
                if isinstance(bs[0], np.ndarray) or isinstance(bs[0], list):
                    if isinstance(bs[0][0], bool):  # statdp svt {T,F}^m
                        aa = np.array([MyBits(bb) for bb in bs])[res>0]
                    elif isinstance(bs[0][0], float):  # statdp real / dpsniper noisyHist
                        aa = np.array([MyReals(bb) for bb in bs])[res>0]
                    elif isinstance(bs[0][0], np.int64):  # dpsniper svt
                        aa = np.array(bs)[res > 0]
                    elif 'rappor' in self.mechanism.__class__.__name__.lower() and isinstance(bs[0][0], np.float64):  # dpsniper *rappor
                        aa = np.array(bs)[res > 0]
                    else:
                        raise NotImplementedError
                elif isinstance(bs[0], float) or isinstance(bs[0], np.float):  # dpsniper real
                    aa = np.array([MyReals(bb) for bb in bs])[res>0]
                else:
                    aa = np.array(bs)[res>0]
                bb = res[res>0]
                if len(aa) > 0:
                    aa, bb = unique(aa, bb)
                log_bs.extend(aa)
                log_bprobs.extend(bb)

            frac_counts += [math.fsum(res)]

        return math.fsum(frac_counts), log_bs, log_bprobs

    def get_variance(self):
        """
        Returns the variance of estimations
        """
        return 1.0/(4.0*self.n_samples)


class EpsEstimator:
    """
    Class for computing an estimate of
        eps(a, a', S) = log(Pr[M(a) in S]) - log(Pr[M(a') in S])
    """

    def __init__(self, pr_estimator: PrEstimator, allow_swap: bool = False):
        """
        Creates an estimator.

        Args:
            pr_estimator: the PrEstimator used to estimate probabilities based on samples
            allow_swap: whether probabilities may be swapped
        """
        self.pr_estimator = pr_estimator
        self.allow_swap = allow_swap

    def compute_eps_estimate(self, a1, a2, attack: Attack) -> (float, float, bool):
        """
        Estimates eps(a2, a2, attack) using samples.

        Returns:
            a tuple (eps, lcb), where eps is the eps estimate and lcb is a lower confidence bound for eps
        """
        swapped = False
        p1 = self.pr_estimator.compute_pr_estimate(a1, attack)
        p2 = self.pr_estimator.compute_pr_estimate(a2, attack)
        log.info("p1=%f, p2=%f", p1, p2)
        log.data("p1", p1)
        log.data("p2", p2)

        if p1 < p2:
            if self.allow_swap:
                p1, p2 = p2, p1
                log.debug("swapped probabilitites p1, p2")
            else:
                log.warning("probability p1 < p2 for eps estimation")

        eps = self._compute_eps(p1, p2)
        lcb = self._compute_lcb(p1, p2)
        return eps, lcb

    @staticmethod
    def _compute_eps(p1, p2):
        if p1 > 0 and p2 == 0:
            eps = float("infinity")
        elif p1 <= 0:
            eps = 0
        else:
            eps = np.log(p1) - np.log(p2)
        return eps

    def _compute_lcb(self, p1, p2):
        n_samples = self.pr_estimator.n_samples
        # confidence accounts for the fact that two bounds could be incorrect (union bound)
        confidence = 1 - (1-self.pr_estimator.config.confidence) / 2
        p1_lcb = lcb(n_samples, int(p1 * n_samples), 1-confidence)
        p2_ucb = ucb(n_samples, int(p2 * n_samples), 1-confidence)
        return self._compute_eps(p1_lcb, p2_ucb)
