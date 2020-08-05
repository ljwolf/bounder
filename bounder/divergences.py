import numpy
from utils import njit
from scipy.spatial import distance
from scipy.special import entr, rel_entr, kl_div


def entropy(p):
    """
    Compute the entropy between rows of arrays, handling the case when p = 0
     as having an entropy of zero. 

    Arguments
    ---------
    p   :   numpy.ndarray of shape (n,k)
        contains probabilities to use in entropy, and will be unit-standardized
        if not already done. 
    q   :   numpy.ndarray of shape (m,k)
        contains probabilities to use in entropy, and will be unit-standardized
        if not already done. If this is not provided, the entropy of the 
        first argument is computed alone. 
    
    Returns
    -------
    numpy.ndarray of shape (n,) containing the entropies of each row of p, or the
    relative entropy of p|q. 
    """
    p = numpy.asarray(p)
    if p.ndim == 1:
        p = p.reshape(1, -1)
    p = p / p.sum(axis=1, keepdims=True)
    core = entr(p)
    core[numpy.isinf(core)] = 0  # ignore log(p) | p==0
    return core.sum(axis=1).squeeze()


def relative_entropy(p, q):
    """
    Compute the relative entropy between rows of arrays, handling the case when p = 0
     as having an entropy of zero. 

    Arguments
    ---------
    p   :   numpy.ndarray of shape (n,k)
        contains probabilities to use in entropy, and will be unit-standardized
        if not already done. 
    q   :   numpy.ndarray of shape (m,k)
        contains probabilities to use in entropy, and will be unit-standardized
        if not already done. 
    
    Returns
    -------
    numpy.ndarray of shape (n,) containing the cross entropies of p | q
    """
    p = numpy.asarray(p)
    q = numpy.asarray(q)
    q = numpy.asarray(q)
    if q.ndim == 1:
        q = q.reshape(1, -1)
    q = q / q.sum(axis=1, keepdims=True)
    core = rel_entr(p, q)
    return core.sum(axis=1).squeeze()


def kullback_leibler(p, q):
    """
    Compute the kullback leibler divergence between rows of arrays, handling the case when p = 0
     as having an entropy of zero. 

    Arguments
    ---------
    p   :   numpy.ndarray of shape (n,k)
        contains probabilities to use in entropy, and will be unit-standardized
        if not already done. 
    q   :   numpy.ndarray of shape (m,k)
        contains probabilities to use in entropy, and will be unit-standardized
        if not already done. 
    
    Returns
    -------
    numpy.ndarray of shape (n,) containing the KL(p | q)
    """
    p = numpy.asarray(p)
    q = numpy.asarray(q)
    if q.ndim == 1:
        q = q.reshape(1, -1)
    if p.ndim == 1:
        p = p.reshape(1, -1)
    p = p / p.sum(axis=1, keepdims=True)
    q = q / q.sum(axis=1, keepdims=True)
    core = kl_div(p, q)
    return core.sum(axis=1).squeeze()


def jensen_shannon(n, m, weighted=True):
    """
    Compute the Jensen-Shannon divergence between rows of arrays.
    NOTE: 

    This function is 'vectorized' over n. So, the result matrix
    is always aligned with the inumpyut on n
    """
    n = numpy.asarray(n)
    m = numpy.asarray(m)
    if not weighted:
        return distance.jensenshannon(n.T, m.T) ** 2
    if m.ndim == 1:
        m = m.reshape(1, -1)
    if n.ndim == 1:
        n = n.reshape(1, -1)
    if n.shape[0] > 1:
        return numpy.array(
            [jensen_shannon(ni, m, weighted=weighted) for ni in n]
        ).squeeze()
    qs = m / m.sum(axis=1, keepdims=True)
    p = n / n.sum()
    n_mass, M_mass, M_masses = n.sum(), m.sum(), m.sum(axis=1)
    M_marginal = m.sum(axis=0)
    r = (n + M_marginal) / (n_mass + M_mass)
    weights = numpy.array([n_mass, *M_masses])
    dists = numpy.row_stack((p, qs))
    R = numpy.row_stack([r] * len(dists))
    return numpy.average(kullback_leibler(dists, R), weights=weights).squeeze()


def cityblock(p, q):
    """
    Compute the cityblock distance (l1 distance) over rows of arrays. 

    See scipy.spatial.distance.cityblock for more information. 
    """
    p = numpy.asarray(p)
    q = numpy.asarray(q)
    assert p.shape == q.shape
    return numpy.sum(numpy.abs(p - q), axis=1)


def chebyshev(p, q):
    """
    Compute the chebyshev distance over rows of arrays. 

    See scipy.spatial.distance.chebyshev for more information. 
    """
    p = numpy.asarray(p)
    q = numpy.asarray(q)
    assert p.shape == q.shape
    return numpy.max(numpy.abs(p - q), axis=1)


def euclidean(p, q):
    """
    Compute the euclidean distance (l2 distance) over rows of arrays. 

    See scipy.spatial.distance.euclidean for more information. 
    """
    p = numpy.asarray(p)
    q = numpy.asarray(q)
    assert p.shape == q.shape
    return numpy.sqrt(numpy.sum((p - q) ** 2, axis=1))


def sqeuclidean(p, q):
    """
    Compute the squared euclidean distance over rows of arrays. 

    See scipy.spatial.distance.sqeuclidean for more information. 
    """
    p = numpy.asarray(p)
    q = numpy.asarray(q)
    assert p.shape == q.shape
    return numpy.sum((p - q) ** 2, axis=1)


_dispatch = dict(
    entropy=entropy,
    relative_entropy=relative_entropy,
    kl=kullback_leibler,
    kullback_leibler=kullback_leibler,
    js=jensen_shannon,
    l1=cityblock,
    cityblock=cityblock,
    l2=euclidean,
    euclidean=euclidean,
    sqeuclidean=sqeuclidean,
)


def get_divergence(div_name):
    """
    Resolve the name of a divergence to the callable in this module, or 
    pass through if the requested divergence is itself a callable function. 
    """
    if callable(div_name):
        return div_name
    else:
        return _dispatch[div_name]
