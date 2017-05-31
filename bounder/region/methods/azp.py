from builtins import super

from ..abstracts import Regionalizer, Mogrifyer
from ..metaheuristics.tabu import Tabu
from .. import utils

class AZP(Tabu, Regionalizer, Mogrifyer):
    def __init__(self, objective, constraints=None, remember=False):
        Regionalizer.__init__(self, objective=objective, 
                                    constraints=constraints, 
                                    remember=remember)
        Tabu.__init__(self, remember=remember)
    
    def fit(self, n_clusters, data, 
            alist=None, w=None, nx=None,
            affinity_matrix=None, **kw):
        ...
