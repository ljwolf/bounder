import numpy
from scipy.spatial import cKDTree
from . import divergence
from . import utils

def _areal_entropy(coordinates, data, 
                   threshold, divergence_func, 
                   query_function,
                   relative=True):
    """
    This is a function to compute the areal entropy. All arguments are documented
    in either entropogram or _build_query_func. This function does *no* validation
    of its input, given that it's intended to run on an inner loop.
    """
    n, k = coordinates.shape
    areal_entropies = numpy.empty((n,1))
    for i in range(n):
        q = query_function(i, threshold)
        if relative:
            me, others = [i], list(set(q).difference(set((i,))))
            if len(others) == 0:
                areal_entropies[i] = 0
                continue
            # population-weighted average of surroundings
            # is the same if you use percentages in the input data
            # but will differ if you use differently-sized elements in data
            # other = data[others].sum(axis=0) # add all others up
            # other /= other.sum() # convert the totals to a simplex
            # other *= data[others].sum(axis=1).mean() # multiply by the "typical size"
            # areal_entropies[i] = divergence(data[me], data[others].sum(axis=0))
            ############### 
            # jsd now accommodates vector - to - matrix formulations:
            if divergence_func == divergence.jensen_shannon:
                areal_entropies[i] = divergence_func(data[me], data[others])
            else:
                areal_entropies[i] = divergence_func(data[me], data[others].sum(axis=0))
        else:
            areal_entropies[i] = divergence(data[q])
    return areal_entropies.squeeze()

def _build_query_func(coordinates, knn=False, kdt=None, distances=None):
    """
    This builds a closure that makes querying for distances simpler. The
    closure takes two arguments, index & a value, and gives the
    observations that are closer than that value. The value can be a
    distance, which then the function returns the observations at or closer
    than that distance to i. The value can also be a rank, which then the function
    returns the observations at or closer than that *k-nearest* neighbor.
    """
    if kdt is None and distances is None:
        kdt = cKDTree(coordinates)
    elif kdt is not None and distances is not None:
        raise Exception('Either KDT or Distances may be provided.')
    if kdt is not None:
        if knn:
            query = lambda index, k: kdt.query(coordinates[index], k=k)[-1]
        else:
            query = lambda index, dist: kdt.query_ball_point(coordinates[index], r=dist)
    elif distances is not None:
        if knn:
            ranks = numpy.argsort(distances, axis=1)
            query = lambda index, k: (ranks[index,] < k).nonzero()[0]
        else:
            query = lambda index, dist: (distances[index,] < dist).nonzero()[0]
    else:
        raise Exception('kd-tree or distances were not properly constructed!')
    return query


def entropogram(coordinates, data,
                distances=None, knn=False,
                metric='euclidean', n_bins=30,
                interval=None, divergence='jsd',
                summarize=numpy.mean, 
                relative=True, standardize=True):
    """
    coordinates :   (n,2) numpy array
                    array containing coordinates to compute the entropogram.
    data        :   (n,k) array
                    array containing the data to compute entropy for. May
                    either be a simplex (i.e. rows sum to 1) or raw values.
    distances   :   (n,n) array (default: None)
                    if a precomputed distance matrix is desired to compute areas,
                    such as in the case of using an isochrone/walking distance,
                    use this argument to provide a distance matrix.
    knn         :   bool (default: False)
                    whether to compute the K-Nearest Neighbor entropogram, which
                    provides the entropy as the surrounding area increases as by
                    adding the next k-nearest observations. By default, this
                    computes the *distance-based* entropogram, which adds all
                    observations within the next diameter increase.
    metric       :  str or callable (default: 'euclidean')
                    the metric used to compute distances in the KDTree. Will be
                    ignored if "distances" is provided.
    n_bins       :  int (default: 30)
                    how many bins over which to compute the entropogram. These are
                    the "steps" between the smallest & largest radii (or knn values),
                    so many bins will result in very fine-grained analyses.
    interval     :  tuple (default: None)
                    the range over which the entropogram will be computed. By
                    default, this will compute the entropogram over the minimum
                    and maximum distances (or, 1 to n if knn=True)
    divergence   :  str or callable (default: 'jsd')
                    the divergence metric to use to compute the entropogram. This
                    can be ('kl', 'js', 'kullback_leibler', 'jensen_shannon',
                            'relative_entropy', 'entropy')
    summarize    :  str or callable (default: numpy.mean)
                    function to use to summarize the entropogram. The entropogram
                    provides a value for each observation of entropy/divergence
                    at *each* distance/knn value from interval[0] to interval[1].
                    By default, this function returns the *mean* of all sites'
                    entropy/divergence values at that distance/knn. If None,
                    the full (n, n_bins) matrix of entropy values is provided.
    relative     :  bool (default: True)
                    whether to compute relative entropy or absolute entropy for
                    each site. By "relative," this means that the site will be
                    *compared* to its surroundings (e.g. KL(site || surrounding)).
                    If *not* relative, then all values in the area will be pooled
                    and the standard entropy value computed. (e.g. entropy([site] + [*surrounding]))
    """
    if not relative:
        divergence = 'entropy'
    divergence_func = utils.resolve_div(divergence)
    if distances is not None:
        metric = 'precomputed'
    if interval is None:
        if metric == 'precomputed':
            mindist, maxdist = utils.get_bounding_distances(distances, metric,
                                                            knn=knn)
        else:
            mindist, maxdist = utils.get_bounding_distances(coordinates, metric,
                                                            knn=knn)
    else:
        mindist, maxdist = interval

    if distances is None:
        kdt = cKDTree(coordinates)
    if summarize is None:
        summarize = lambda d, axis: d
    elif isinstance(summarize, str):
        summarize = getattr(numpy, summarize)

    N,k = coordinates.shape
    assert k==2, 'Entropogram is only well defined for 2-dimensional data'
    entropogram = numpy.empty((N,n_bins))
    query = _build_query_func(coordinates, knn=knn, kdt=kdt, distances=distances)
    if knn:
        support = numpy.linspace(mindist, maxdist, num=n_bins).round(0)
    else:
        support = numpy.linspace(mindist, maxdist, num=n_bins)
    for i_d, chunk in enumerate(support):
        entropogram[:,i_d] = _areal_entropy(coordinates, data, chunk, divergence_func,
                                            query_function=query,
                                            relative=relative)
    return summarize(entropogram, axis=0)

def lag_entropy(connectivity, D, divergence=divergence.entropy):
    divergence_func = utils.resolve_div(divergence)

    assert (connectivity.data == 1).all()
    N,p = D.shape
    assert connectivity.shape[0] == connectivity.shape[1] == N
    R = connectivity * D
    raise NotImplementedError('Needs to change to accommodate pop-weighted lag')
    out = numpy.empty((N,))
    for i in range(N):
        out[i] = divergence_func(D[i], R[i])
    return out

def lag_jsd(connectivity, D):
    return lag_entropy(connectivity, D, divergence=divergence.jensen_shannon)
