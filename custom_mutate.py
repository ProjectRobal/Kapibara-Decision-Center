import random
from typing import List,Tuple
from numba import njit
import numba
import numpy as np


ind_type = List[float]

@njit
def mate_params(ind1: ind_type, ind2: ind_type,
                max_ratio: float=0.5) -> Tuple[ind_type, ind_type]:
    """Mate two parameter lists to a sampled ratio.

    Arguments:
        ind1, ind2: Individuals to mate.
        max_ratio: Maximum ratio (uniform sample from 0.0).

    Returns:
        Blended individuals.
    """
    _ind1=np.array(ind1,dtype=np.float32)
    _ind2=np.array(ind2,dtype=np.float32)

    rat = random.uniform(0.0, max_ratio)
    rem = 1 - rat

    o1=rem*_ind1+rat*_ind2
    o2=rat*_ind1+rat*_ind2

    for i, (v1, v2) in enumerate(zip(ind1, ind2)):
        ind1[i] = rem * v1 + rat * v2
        ind2[i] = rat * v1 + rem * v2

    return o1.tolist(), o2.tolist()


@njit
def mutate_params(indpb: float, sigma: ind_type,
                  lbs: ind_type, ubs: ind_type, ind: ind_type) -> Tuple[ind_type,]:
    """Gaussian mutation on a parameter set.

    Arguments:
        indpb: Independent probability for each attribute to be mutated.
        sigma: Standard deviations for the gaussian addition mutation.
        lbs: Lower bounds on entries in individual.
        ubs: Upper bounds on entries in individual.
        ind: Individual to mutate.

    Returns:
        A tuple of one individual.
    """

    for i, s in enumerate(sigma):
        if random.random() < indpb:
            ind[i] += random.gauss(0.0, s)
            if ind[i] < lbs[i]:
                ind[i] = lbs[i]
            elif ind[i] > ubs[i]:
                ind[i] = ubs[i]

    return ind,

