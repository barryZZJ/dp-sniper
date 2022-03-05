import os
import re
import sys
import numpy as np
import json

from witness import Witness
from dataLoader import DataLoader


class MyLogParser:
    def __init__(self, logfile):
        self.logfile = logfile
        self.dAlgs = {}

        with open(self.logfile) as f:
            for line in f:
                dAlg = json.loads(line)
                alg = self.get_universal_alg_name(dAlg['alg'][0])
                d = self.dAlgs.setdefault(alg, {})
                del dAlg['alg']
                for k, v in dAlg.items():
                    if k == 'statdp_temp_result':
                        d.setdefault('a1', []).append(v['a1'])
                        d.setdefault('a2', []).append(v['a2'])
                        d.setdefault('s', []).append(v['event'])
                        d.setdefault('eps', []).append(v['eps'])
                        d.setdefault('lcb', []).append(v['lower_bound'])
                        d.setdefault('pp', []).append(v['postprocessing'])
                    elif k in ['p1', 'p2']:
                        d.setdefault(k, []).append(v)

    def get_witness(self, decimals=3):
        for alg, d in self.dAlgs.items():
            a1 = self.round(d['a1'], decimals)
            a2 = self.round(d['a2'], decimals)
            s = self.round(d['s'], decimals)
            p1 = self.round(d['p1'], decimals)
            p2 = self.round(d['p2'], decimals)
            eps = d['eps']
            eps = np.array(eps, dtype=np.float64)
            eps[np.isnan(eps)] = 0
            eps = self.round(eps, decimals)
            lcb = d['lcb']
            lcb = np.array(lcb, dtype=np.float64)
            lcb[np.isnan(lcb)] = 0
            lcb = self.round(lcb, decimals)
            for i in range(len(a1)):
                yield alg, Witness(a1[i], a2[i], s[i], eps[i], 'statdpwrapper', p1=p1[i], p2=p2[i],lcb=lcb[i])

    @staticmethod
    def round(arr, decimals):
        return np.round(arr, decimals).tolist()

    def get_universal_alg_name(self, name):
        name_map = {
            'NoisyHist1': 'NoisyHistogram1',
            'NoisyHist2': 'NoisyHistogram2',
            'ReportNoisyMax1': 'ReportNoisyMax1-Lap',
            'ReportNoisyMax2': 'ReportNoisyMax2-Exp',
            'ReportNoisyMax3': 'ReportNoisyMax3-Lap',
            'ReportNoisyMax4': 'ReportNoisyMax4-Exp',
            'SparseVectorTechnique1': 'SVT1',
            'SparseVectorTechnique2': 'SVT2',
            'SparseVectorTechnique3': 'SVT3',
            'SparseVectorTechnique4': 'SVT4',
            'SparseVectorTechnique5': 'SVT5',
            'SparseVectorTechnique6': 'SVT6',
            'Rappor': 'RAPPOR',
            'OneTimeRappor': 'OneTimeRAPPOR',
            'TruncatedGeometricMechanism': 'TruncatedGeometric',
        }
        uni_name = name_map.get(name)
        if uni_name is None:
            return name
        return uni_name

class StatDPWarpperLoader(DataLoader):
    def __init__(self):
        super(StatDPWarpperLoader, self).__init__()
        basePath = '/media/barry/Data/DPDisprover/dp-sniper'
        self.logsdir = os.path.join(basePath, 'statdpwrapper/experiments/logs')

    def load_data(self):
        for dir, dirnames, filenames in os.walk(self.logsdir):
            for filename in filenames:
                if filename.endswith('data.log'):
                    p = MyLogParser(os.path.join(dir, filename))
                    for alg, wit in p.get_witness():
                        self._push(alg, wit)


if __name__ == '__main__':
    from dataVisualizer import DataVisualizer
    dl = StatDPWarpperLoader()
    dl.load_data()
    vi = DataVisualizer(dl)
    vi.to_excel(filename='statdpwrapper.xls')
