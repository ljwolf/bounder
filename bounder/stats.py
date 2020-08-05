from libpysal import weights
import pandas
from scipy import sparse


def _cast_to_alist(w):
    if isinstance(w, weights.W):
        alist = w.to_adjlist()
    elif sparse.issparse(w):
        row, col = w.nonzero()
        weight = w.data
        alist = pandas.DataFrame.from_dict(dict(focal=row, neighbor=col, weight=weight))
    elif isinstance(w, pandas.DataFrame):
        alist = w
    return alist


def local_geary(X, w, metric="sqeuclidean", permutations=999):
    """
    Compute the local geary statistic for each site.
    """
    alist = _cast_to_alist(w)
    tmp_alist = alist.merge(X, left_on="focal", right_index=True, how="left").merge(
        X,
        left_on="neighbor",
        right_index=True,
        how="left",
        suffixes=("_focal", "_neighbor"),
    )
    focals, neighbors = (
        tmp_alist.filter(like="_focal"),
        tmp_alist.filter(like="_neighbor"),
    )

    div_func = get_metric(metric)

    diffs = div_func(focals, neighbors)
