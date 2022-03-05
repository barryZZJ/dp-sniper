import logging
from typing import Tuple

from statdpwrapper.my_generate_counterexample import detect_counterexample

from statdpwrapper.postprocessing import PostprocessingConfig, get_postprocessed_algorithms, compose_postprocessing, \
    the_zero_noise_prng
from dpsniper.utils.my_logging import log, time_measure
from dpsniper.probability.estimators import EpsEstimator
from statdpwrapper.verification import StatDPAttack


class BinarySearch:
    """
    Helper class for transforming StatDP to a maximizing-power approach using binary search.
    """

    def __init__(self, algorithm, num_input, sensitivity, detect_iterations, default_kwargs, pp_config: PostprocessingConfig, pr_estimator):
        self.algorithm = algorithm
        self.default_kwargs = default_kwargs
        self.num_input = num_input
        self.sensitivity = sensitivity
        self.detect_iterations = detect_iterations
        self.epsEstimator = EpsEstimator(pr_estimator, allow_swap=True)

        self.all_postprocessed_algs = get_postprocessed_algorithms(algorithm, pp_config)
        self._nof_probes = 0  # No. of probes

    def compute_eps_estimate(self, a1, a2, event, postprocessing, a0):
        attack = StatDPAttack(event, postprocessing)
        if postprocessing.requires_noisefree_reference:
            noisefree_reference = self.algorithm(the_zero_noise_prng, a0, **self.default_kwargs)
            attack.set_noisefree_reference(noisefree_reference)
        eps_verified, eps_lcb, swapped = self.epsEstimator.compute_eps_estimate(a1, a2, attack)
        return eps_verified, eps_lcb, (a1, a2) if not swapped else (a2, a1)

    def find(self, p_value_threshold: float, precision: float) -> Tuple:
        """
        Returns the tuple (epsilon, a1, a2, event, postprocessing, a0) with highest epsilon for which the p_value is
        still below p_value_threshold, up to a given precision. Here, a0 is the reference input to be used for
        HammingDistance postprocessing.
        """
        self._nof_probes = 0
        left, right = self._exponential_init(p_value_threshold)  # left/right = (eps, attack=(a1, a2, event, postprocessing, a0) )
        log.info("bounds for eps: [%f, %f]", left[0], right[0])

        res = self._binary_search(left, right, p_value_threshold, precision)
        log.info("required %d probes", self._nof_probes)
        log.data("statdp_nof_probes", self._nof_probes)
        return res

    def _probe(self, eps) -> Tuple:
        """
        Returns a tuple (p_value, (a1, a2, event, postprocessing, a0))

        a0 is the reference input to be used for HammingDistance postprocessing
        """
        log.info("checking eps = %f", eps)
        self._nof_probes += 1
        with time_measure("statdp_time_one_probe"):
            min_p_value = 1.0
            min_attack = None
            for pps in self.all_postprocessed_algs:
                log.info("trying postprocessing %s...", str(pps))
                self.default_kwargs['alg'] = pps
                result = detect_counterexample(compose_postprocessing,
                                               eps,
                                               num_input=self.num_input,
                                               default_kwargs=self.default_kwargs,
                                               sensitivity=self.sensitivity,
                                               detect_iterations=self.detect_iterations,
                                               quiet=True,
                                               loglevel=logging.INFO)
                del self.default_kwargs['alg']
                (_, p_value, d1, d2, kwargs, event) = result[0]
                if min_attack is None or p_value < min_p_value:
                    min_p_value = p_value
                    min_attack = (d1, d2, event, pps.postprocessing, kwargs['_d1'])

            log.info("p_value = %f", min_p_value)
            log.info("event = %s", min_attack)
            log.data("statdp_intermediate_probe", {"eps": eps, "p_value": min_p_value})
            return min_p_value, min_attack

    def _exponential_init(self, p_value_threshold: float):
        log.info("running exponential search")
        eps = 0.005
        p_value = 0.0
        prev_p_value = 0.0
        attack = None
        prev_attack = None
        while p_value < p_value_threshold:
            prev_p_value = p_value
            prev_attack = attack
            eps *= 2
            p_value, attack = self._probe(eps)

        # record temp witness:
        log_eps, log_eps_lcb, (log_a1, log_a2) = self.compute_eps_estimate(prev_attack[0], prev_attack[1], prev_attack[2], prev_attack[3], prev_attack[4])
        log.data("statdp_temp_result", {"eps": log_eps,
                                        "lower_bound": log_eps_lcb,
                                        "p_value": prev_p_value,
                                        "a1": log_a1,
                                        "a2": log_a2,
                                        "event": prev_attack[2],
                                        "postprocessing": str(prev_attack[3])})
        return (eps/2, prev_attack), (eps, attack)

    def _binary_search(self, left_tup, right_tup, p_value_threshold: float, precision: float):
        left = left_tup[0]    # eps
        right = right_tup[0]  # eps
        left_attack = left_tup[1]  # (a1, a2, event, postprocessing, a0)

        # invariant: p_value at left is strictly below p_value_threshold
        while right - left > precision:

            mid = left + ((right - left) / 2)
            p_value, mid_attack = self._probe(mid)
            if p_value < p_value_threshold:
                left = mid
                left_attack = mid_attack
                # record temp witness:
                log_eps, log_eps_lcb, (log_a1, log_a2) = self.compute_eps_estimate(left_attack[0], left_attack[1],
                                                                                   left_attack[2], left_attack[3],
                                                                                   left_attack[4])
                log.data("statdp_temp_result", {"eps": log_eps,
                                                "lower_bound": log_eps_lcb,
                                                "p_value": p_value,
                                                "a1": log_a1,
                                                "a2": log_a2,
                                                "event": left_attack[2],
                                                "postprocessing": str(left_attack[3])})
            else:
                right = mid
        log.info("finished binary search")
        log.info("  eps = %f", left)
        log.info("  attack = %s", left_attack)
        return left, left_attack[0], left_attack[1], left_attack[2], left_attack[3], left_attack[4]
