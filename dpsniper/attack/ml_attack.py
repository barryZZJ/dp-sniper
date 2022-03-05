from dpsniper.attack.attack import Attack
from dpsniper.classifiers.stable_classifier import StableClassifier

import numpy as np
import os


class MlAttack(Attack):
    """
    An attack based on membership inference.
    """

    def __init__(self, classifier: StableClassifier, thresh: float, q: float):
        """
        Create an attack.

        Args:
            classifier: trained membership inference classifier
            thresh: probability threshold
            q: threshold tie-break probability
        """
        self.classifier = classifier
        self.thresh = thresh
        self.q = q

    def check(self, b):
        if len(b.shape) == 1:
            # make sure b has shape (n_samples, 1)
            b = np.atleast_2d(b).T

        probs = self.classifier.predict_probabilities(b)
        above_thresh_probs = self._compute_above_thresh_probs(probs)
        return above_thresh_probs

    def _compute_above_thresh_probs(self, classifier_probs):
        above = (classifier_probs > self.thresh).astype(float)    # 1.0 iff classifier_probs > t
        equal = (classifier_probs == self.thresh).astype(float)   # 1.0 iff classifier_probs == t
        return above + self.q * equal

    def __str__(self):
        if self.classifier.state_dict_file is None:
            return "t = {}, q = {}, CLASSIFIER = {}".format(self.thresh, self.q, str(self.classifier))
        return "t = {}, q = {}, CLASSIFIER = {}, file = {}".format(self.thresh, self.q, str(self.classifier), os.path.basename(self.classifier.state_dict_file))

    def to_json(self):
        d = {
            't': self.thresh,
            'q': self.q,
            'CLSSIFIER': str(self.classifier),
            'file': os.path.basename(self.classifier.state_dict_file) if self.classifier.state_dict_file is not None else None
        }
        return d