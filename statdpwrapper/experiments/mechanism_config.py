from statdp import ONE_DIFFER, ALL_DIFFER

from statdpwrapper.algorithms_ext import *
from statdpwrapper.algorithms import *
from statdpwrapper.postprocessing import PostprocessingConfig

statdp_mechanism_map = {}
statdp_arguments_map = {}
statdp_postprocessing_map = {}
statdp_num_inputs_map = {}
statdp_sensitivity_map = {}


def register(name, mechanism, pp_config: PostprocessingConfig, num_inputs: tuple, sensitivity, arguments: dict):
    statdp_mechanism_map[name] = mechanism
    statdp_arguments_map[name] = arguments
    statdp_postprocessing_map[name] = pp_config
    statdp_num_inputs_map[name] = num_inputs
    statdp_sensitivity_map[name] = sensitivity

# num_num_inputs 有两个元素相当于statDP跑两次
register("LaplaceMechanism",
         laplace_mechanism,
         PostprocessingConfig(
             is_numerical=True,
             is_categorical=False,
             has_variable_dimensions=False,
             max_dimensions=1),
         num_inputs=(1, 1),
         sensitivity=ONE_DIFFER,
         arguments={'epsilon': 0.1})
register("TruncatedGeometricMechanism",
         truncated_geometric,
         PostprocessingConfig(
             is_numerical=True,
             is_categorical=False,
             has_variable_dimensions=False,
             max_dimensions=1),
         num_inputs=(1, 1),
         sensitivity=ONE_DIFFER,
         arguments={'epsilon': 0.1, 'n': 5})

register("NoisyHist1",
         histogram,
         PostprocessingConfig(
             is_numerical=True,
             is_categorical=False,
             has_variable_dimensions=False,
             max_dimensions=5),
         num_inputs=(5, 5),
         sensitivity=ONE_DIFFER,
         arguments={'epsilon': 0.1})
register("NoisyHist2",
         histogram_eps,
         PostprocessingConfig(
             is_numerical=True,
             is_categorical=False,
             has_variable_dimensions=False,
             max_dimensions=5),
         num_inputs=(5, 5),
         sensitivity=ONE_DIFFER,
         arguments={'epsilon': 0.1})

register("ReportNoisyMax1",
         noisy_max_v1a,
         PostprocessingConfig(
             is_numerical=True,
             is_categorical=False,
             has_variable_dimensions=False,
             max_dimensions=1),
         num_inputs=(5, 5),
         sensitivity=ALL_DIFFER,
         arguments={'epsilon': 0.1})
register("ReportNoisyMax2",
         noisy_max_v2a,
         PostprocessingConfig(
             is_numerical=True,
             is_categorical=False,
             has_variable_dimensions=False,
             max_dimensions=1),
         num_inputs=(5, 5),
         sensitivity=ALL_DIFFER,
         arguments={'epsilon': 0.1})
register("ReportNoisyMax3",
         noisy_max_v1b,
         PostprocessingConfig(
             is_numerical=True,
             is_categorical=False,
             has_variable_dimensions=False,
             max_dimensions=1),
         num_inputs=(5, 5),
         sensitivity=ALL_DIFFER,
         arguments={'epsilon': 0.1})
register("ReportNoisyMax4",
         noisy_max_v2b,
         PostprocessingConfig(
             is_numerical=True,
             is_categorical=False,
             has_variable_dimensions=False,
             max_dimensions=1),
         num_inputs=(5, 5),
         sensitivity=ALL_DIFFER,
         arguments={'epsilon': 0.1})

register("SparseVectorTechnique1",
         SVT,
         PostprocessingConfig(
             is_numerical=False,
             is_categorical=True,
             has_variable_dimensions=True,
             max_dimensions=10,
             categories=[True, False]),

         num_inputs=(10, 10),
         sensitivity=ALL_DIFFER,
         arguments={'epsilon': 0.1,
                    'N': 1,
                    'T': 0.5})
register("SparseVectorTechnique2",
         SVT2,
         PostprocessingConfig(
             is_numerical=False,
             is_categorical=True,
             has_variable_dimensions=True,
             max_dimensions=10,
             categories=[True, False]),

         num_inputs=(10, 10),
         sensitivity=ALL_DIFFER,
         arguments={'epsilon': 0.1,
                    'N': 1,
                    'T': 1})
register("SparseVectorTechnique3",
         iSVT4,
         PostprocessingConfig(
             is_numerical=True,
             is_categorical=True,
             has_variable_dimensions=True,
             max_dimensions=10,
             categories=[False]),

         num_inputs=(10, 10),
         sensitivity=ALL_DIFFER,
         arguments={'epsilon': 0.1,
                    'N': 1,
                    'T': 1})
register("SparseVectorTechnique4",
         iSVT3,
         PostprocessingConfig(
             is_numerical=False,
             is_categorical=True,
             has_variable_dimensions=True,
             max_dimensions=10,
             categories=[True, False]),

         num_inputs=(10, 10),
         sensitivity=ALL_DIFFER,
         arguments={'epsilon': 0.1,
                    'N': 1,
                    'T': 1})
register("SparseVectorTechnique5",
         iSVT1,
         PostprocessingConfig(
             is_numerical=False,
             is_categorical=True,
             has_variable_dimensions=False,
             max_dimensions=10,
             categories=[True, False]),

         num_inputs=(10, 10),
         sensitivity=ALL_DIFFER,
         arguments={'epsilon': 0.05,
                    'N': 1,
                    'T': 1})
register("SparseVectorTechnique6",
         iSVT2,
         PostprocessingConfig(
             is_numerical=False,
             is_categorical=True,
             has_variable_dimensions=False,
             max_dimensions=10,
             categories=[True, False]),

         num_inputs=(10, 10),
         sensitivity=ALL_DIFFER,
         arguments={'epsilon': 0.1,
                    'N': 1,
                    'T': 1})

register("Rappor",
         rappor,
         PostprocessingConfig(
             is_numerical=False,
             is_categorical=True,
             has_variable_dimensions=False,
             max_dimensions=10,
             categories=[1.0, 0.0]),
         num_inputs=(1, 1),
         sensitivity=ONE_DIFFER,
         arguments={'epsilon': 0.0,
                    'n_hashes': 4,
                    'filter_size': 20,
                    'f': 0.75,
                    'p': 0.45,
                    'q': 0.55})
register("OneTimeRappor",
         one_time_rappor,
         PostprocessingConfig(
             is_numerical=False,
             is_categorical=True,
             has_variable_dimensions=False,
             max_dimensions=10,
             categories=[1.0, 0.0]),
         num_inputs=(1, 1),
         sensitivity=ONE_DIFFER,
         arguments={'epsilon': 0.0,
                    'n_hashes': 4,
                    'filter_size': 20,
                    'f': 0.95})

register("LaplaceParallel",
         laplace_parallel,
         PostprocessingConfig(
             is_numerical=True,
             is_categorical=False,
             has_variable_dimensions=False,
             max_dimensions=20),
         num_inputs=(1, 1),
         sensitivity=ONE_DIFFER,
         arguments={'epsilon': 0.005,
                    'n_parallel': 20})

register("SVT34Parallel",
         svt_34_parallel,
         PostprocessingConfig(
             is_numerical=True,
             is_categorical=True,
             has_variable_dimensions=False,
             max_dimensions=20,
             categories=[True, False, None]),
         num_inputs=(10, 10),
         sensitivity=ALL_DIFFER,
         arguments={'epsilon': 0.1,
                    'N': 2,
                    'T': 1})
register("PrefixSum",
         prefix_sum,
         PostprocessingConfig(
             is_numerical=True,
             is_categorical=False,
             has_variable_dimensions=False,
             max_dimensions=10),
         num_inputs=(10, 10),
         sensitivity=ALL_DIFFER,
         arguments={'epsilon': 0.1})
register("NumericalSVT",
         numerical_svt,
         PostprocessingConfig(
             is_numerical=True,
             is_categorical=False,
             has_variable_dimensions=True,
             max_dimensions=10),
         num_inputs=(10, 10),
         sensitivity=ALL_DIFFER,
         arguments={'epsilon': 0.1,
                    'N': 2,
                    'T': 1})
