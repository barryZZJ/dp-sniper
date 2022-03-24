import datetime
import os
import socket
from dpsniper.search.ddconfig import DDConfig

from dpsniper.utils.my_multiprocessing import initialize_parallel_executor
from dpsniper.utils.paths import get_output_directory, set_output_directory
from statdpwrapper.experiments.mechanism_config import statdp_mechanism_map, statdp_postprocessing_map, \
    statdp_num_inputs_map, statdp_sensitivity_map, statdp_arguments_map

from dpsniper.utils.my_logging import log, log_context
from dpsniper.probability.estimators import EpsEstimator
from statdpwrapper.verification import StatDPAttack, StatDPPrEstimator
from statdpwrapper.postprocessing import the_zero_noise_prng

from statdpOutSample.run import runs


def sample_output(alg_name, a1, a2, event, postprocessing, a0, n_processes: int, out_dir: str):

    # with initialize_parallel_executor(n_processes, out_dir):
    config = DDConfig()
    mechanism = statdp_mechanism_map[alg_name]
    kwargs = statdp_arguments_map[alg_name]

    pr_estimator = StatDPPrEstimator(mechanism,
                                     # 10000,
                                     config.n_check,
                                     config,
                                     use_parallel_executor=False,
                                     log_outputs=True,
                                     **kwargs)

    epsEstimator = EpsEstimator(pr_estimator, allow_swap=True)
    # record temp witness:
    log.info("sampling...")

    attack = StatDPAttack(event, postprocessing)
    if postprocessing.requires_noisefree_reference:
        noisefree_reference = mechanism(the_zero_noise_prng, a0, **kwargs)
        attack.set_noisefree_reference(noisefree_reference)
    eps_verified, eps_lcb = epsEstimator.compute_eps_estimate(a1, a2, attack)

    log.data("statdp_temp_result", {"eps": eps_verified,
                                    "lower_bound": eps_lcb,
                                    "a1": a1,
                                    "a2": a2,
                                    "event": event,
                                    "postprocessing": str(postprocessing)})
    log.info("finished")


if __name__ == "__main__":
    n_processes = 12

    for i, run in enumerate(runs[:32]):
        timestamp = '{:%Y-%m-%d_%H-%M-%S}'.format(datetime.datetime.now())
        hostname = socket.gethostname()
        log_tag = timestamp + '_' + hostname
        basepath = os.path.dirname(__file__)
        log_dir = os.path.join(basepath, "logs", log_tag)
        with log_context(run.name):

            log.configure("WARNING")
            set_output_directory(log_dir)
            logs_dir = get_output_directory("logs")
            log_file = os.path.join(logs_dir, "statdp_{}_log.log".format(run.postfix))
            data_file = os.path.join(logs_dir, "statdp_{}_data.log".format(run.postfix))

            if os.path.exists(log_file):
                log.warning("removing existing log file '%s'", log_file)
                os.remove(log_file)
            if os.path.exists(data_file):
                log.warning("removing existing log file '%s'", data_file)
                os.remove(data_file)

            log.configure("INFO", log_file=log_file, data_file=data_file, file_level="INFO")

            log.info("run %d, mech %s", i, run.name)
            sample_output(run.name, run.a1, run.a2, run.event, run.pp, run.a0, n_processes, out_dir=log_dir)

        for dir, _, filenames in os.walk(log_dir):
            for filename in filenames:
                if 'proc' in filename:
                    os.remove(os.path.join(dir, filename))