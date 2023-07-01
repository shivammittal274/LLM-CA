from numpy.linalg import norm
from typing import Dict, Tuple, Any

import numpy as np
from numpy import dot
import settings

from gptcache.similarity_evaluation import SimilarityEvaluation


def calculateCosSim(a, b):
    cos_sim = dot(a, b)/(norm(a)*norm(b))
    return cos_sim

class CosineSimEvaluation(SimilarityEvaluation):
    def __init__(self, threshold = 0.95):
        self.threshold = threshold

    def evaluation(
        self, src_dict: Dict[str, Any], cache_dict: Dict[str, Any], **_
    ) -> float:

        src_embedding = src_dict['embedding']
        cache_embedding = cache_dict['embedding']

        score = calculateCosSim(src_embedding, cache_embedding)
        settings.cache_hit = max(score>self.threshold,  settings.cache_hit)
        return score>self.threshold


    def range(self) -> Tuple[float, float]:
        return 0.0, 1.0
