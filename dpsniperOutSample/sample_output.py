import glob

import datetime
import socket
import os
from dpsniper.utils.paths import set_output_directory, get_output_directory

from dpsniper.utils.my_multiprocessing import initialize_parallel_executor

from dpsniper.probability.estimators import PrEstimator, EpsEstimator

from dpsniper.search.ddsearch import DDConfig
from dpsniper.search.ddwitness import DDWitness
from dpsniper.utils.torch import torch_initialize
from dpsniper.utils.my_logging import log, log_context

from run import runs

def sample_output(mechanism, a1, a2, attack):
    config = DDConfig(n_processes=n_processes)
    pr_estimator = EpsEstimator(PrEstimator(mechanism,
                                            # 1000,
                                            config.n_check,
                                            config,
                                            False,
                                            log_outputs=True))
    cur = DDWitness(a1, a2, attack)
    log.info("computing estimate for eps...")
    cur.compute_eps_using_estimator(pr_estimator)
    log.data("result", cur.to_json())
    log.info("done!")



if __name__ == "__main__":
    n_processes = 12
    torch_threads = 8
    torch_device = 'cpu'
    torch_initialize(torch_threads, torch_device)
    log.configure("INFO")

    for i, run in enumerate(runs):
        timestamp = '{:%Y-%m-%d_%H-%M-%S}'.format(datetime.datetime.now())
        hostname = socket.gethostname()
        log_tag = timestamp + '_' + hostname + '-out'
        log_dir = os.path.join("logs", log_tag)

        with log_context(run.name):

            set_output_directory(log_dir)
            logs_dir = get_output_directory("logs")

            for fname in glob.glob(os.path.join(logs_dir, run.series_name + "_*.log")):
                log.warning("removing existing log file '%s'", fname)
                os.remove(fname)

            log_file = os.path.join(logs_dir, "{}_log.log".format(run.series_name))
            data_file = os.path.join(logs_dir, "{}_data.log".format(run.series_name))
            log.configure("INFO", log_file=log_file, file_level='INFO', data_file=data_file)

            # with initialize_parallel_executor(n_processes, log_dir):
            log.info("run %d, mech %s", i, run.name)
            sample_output(run.mechanism, run.a1, run.a2, run.attack)