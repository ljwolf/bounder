from libpysal.weights.adjtools import _get_W_and_alist as get_w_and_alist
from libpysal.weights.util import get_points_array
from sklearn.preprocessing import normalize as sknormalize
from networkx import NetworkXPointlessConcept
import networkx as nx

def unit_norm(X, axis=0):
    return X / X.max(axis=axis)

def connected_or_null(x):
    try:    
        return nx.is_connected(self.nx.subgraph(x.index))
    except NetworkXPointlessConcept:
        return True
