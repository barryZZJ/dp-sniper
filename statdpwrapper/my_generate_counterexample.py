"""
modification of statdp.detect_counterexample storing the first input

MIT License, Copyright (c) 2018-2019 Yuxin Wang
"""
import logging
import multiprocessing as mp

import tqdm

from statdp.generators import generate_arguments, generate_databases, ALL_DIFFER, ONE_DIFFER
from statdp.hypotest import hypothesis_test
from statdp.selectors import select_event

logger = logging.getLogger(__name__)


def detect_counterexample(algorithm, test_epsilon, default_kwargs=None, databases=None, num_input=(5, 10),
                          event_iterations=100000, detect_iterations=500000, cores=None, sensitivity=ALL_DIFFER,
                          quiet=False, loglevel=logging.INFO):
    """
    :param algorithm: The algorithm to test for.
    :param test_epsilon: The privacy budget to test for, can either be a number or a tuple/list.
    :param default_kwargs: The default arguments the algorithm needs except the first Queries argument.
    :param databases: The databases to run for detection, optional.
    :param num_input: The length of input to generate, not used if database param is specified.
    :param event_iterations: The iterations for event selector to run.
    :param detect_iterations: The iterations for detector to run.
    :param cores: The number of max processes to set for multiprocessing.Pool(), os.cpu_count() is used if None.
    :param sensitivity: The sensitivity setting, all queries can differ by one or just one query can differ by one.
    :param quiet: Do not print progress bar or messages, logs are not affected.
    :param loglevel: The loglevel for logging package.
    :return: [(epsilon, p, d1, d2, kwargs, event)] The epsilon-p pairs along with databases/arguments/selected event.
    """
    # initialize an empty default kwargs if None is given
    default_kwargs = default_kwargs if default_kwargs else {}

    logging.basicConfig(level=loglevel)
    logger.info(f'Start detection for counterexample on {algorithm.__name__} with test epsilon {test_epsilon}')
    logger.info(f'Options -> default_kwargs: {default_kwargs} | databases: {databases} | cores:{cores}')

    input_list = []
    if databases is not None:
        d1, d2 = databases
        kwargs = generate_arguments(algorithm, d1, d2, default_kwargs=default_kwargs)
        input_list = ((d1, d2, kwargs),)
    else:
        num_input = (int(num_input), ) if isinstance(num_input, (int, float)) else num_input
        for num in num_input:
            input_list.extend(
                generate_databases(algorithm, num, default_kwargs=default_kwargs, sensitivity=sensitivity))

    # ------------ BEGIN EDITS ------------
    new_input_list = []
    for db in input_list:
        d1, d2, kwargs = db
        new_kwargs = kwargs.copy()
        # remember the first input (for HammingDistance postprocessing)
        new_kwargs['_d1'] = d1
        new_input_list.append((d1, d2, new_kwargs))
    input_list = new_input_list
    # ------------ END EDITS ------------

    result = []

    # convert int/float or iterable into tuple (so that it has length information)
    test_epsilon = (test_epsilon, ) if isinstance(test_epsilon, (int, float)) else test_epsilon

    with mp.Pool(cores) as pool:
        for _, epsilon in tqdm.tqdm(enumerate(test_epsilon), total=len(test_epsilon), unit='test', desc='Detection',
                                    disable=quiet):
            d1, d2, kwargs, event = select_event(algorithm, input_list, epsilon, event_iterations, quiet=quiet,
                                                 process_pool=pool)
            p = hypothesis_test(algorithm, d1, d2, kwargs, event, epsilon, detect_iterations, report_p2=False,
                                process_pool=pool)
            result.append((epsilon, float(p), d1, d2, kwargs, event))
            if not quiet:
                tqdm.tqdm.write(f'Epsilon: {epsilon} | p-value: {p:5.3f} | Event: {event}')
            logger.info(f'D1: {d1} | D2: {d2} | kwargs: {kwargs}')

        return result
