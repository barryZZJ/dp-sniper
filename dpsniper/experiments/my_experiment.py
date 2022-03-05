from dpsniper.attack.dpsniper import DPSniper
from dpsniper.experiments.experiment_runner import BaseExperiment
from dpsniper.utils.my_logging import time_measure, log
from dpsniper.search.ddsearch import DDSearch


class MyExperiment(BaseExperiment):
    def __init__(self, name, mechanism, input_pair_sampler, classifier_factory, config):
        super().__init__(name)
        self.mechanism = mechanism
        self.input_pair_sampler = input_pair_sampler
        self.classifier_factory = classifier_factory
        self.config = config

    def run(self):
        log.info("using configuration %s", self.config)
        log.data("config", self.config.__dict__)

        attack_opt = DPSniper(self.mechanism, self.classifier_factory, self.config)

        with time_measure("time_dd_search"):
            log.debug("running dd-search...")
            opt = DDSearch(self.mechanism, attack_opt, self.input_pair_sampler, self.config)
            res = opt.run()
        log.info("finished dd-search, preliminary eps=%f", res.eps)

        log.warning("Skipped estimating eps with high precision to save time")

        # with time_measure("time_final_estimate_eps"):
        #     log.debug("computing final eps estimate...")
        #     res.compute_eps_high_precision(self.mechanism, self.config)
        #
        # log.info("done!")
        # log.info("> a1      = {}".format(res.a1))
        # log.info("> a2      = {}".format(res.a2))
        # log.info("> attack  = {}".format(res.attack))
        # log.info("> eps     = {}".format(res.eps))
        # log.info("> eps lcb = {}".format(res.lower_bound))
        #
        # log.data("best_result", res.to_json())

