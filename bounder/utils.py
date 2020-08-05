import numpy, pandas

try:
    from numba import njit, jit, prange, boolean
except (ImportError, ModuleNotFoundError):

    def jit(*dec_args, **dec_kwargs):
        """
        decorator mimicking numba.jit
        """

        def intercepted_function(f, *f_args, **f_kwargs):
            return f

        return intercepted_function

    njit = jit

    prange = range
    boolean = bool


def resolve_div(metric):
    if isinstance(metric, str):
        try:
            from . import divergence

            metric = getattr(divergence, metric)
        except AttributeError:
            if metric.lower().startswith("js"):
                metric = divergence.jensen_shannon
            elif metric.lower().startswith("kl"):
                metric = divergence.kullback_leibler
            elif metric.lower().startswith("re"):
                metric = divergence.relative_entropy
            else:
                raise KeyError('Metric "{}" not understood.'.format(metric))
    elif callable(metric):
        pass
    else:
        raise KeyError(
            'Metric "{}" not understood. '
            'Please provide a string in ("entropy", "jsd") '
            "or a function".format(metric)
        )
    return metric


def get_bounding_distances(c, distance_metric, knn=False):
    from scipy.spatial import distance

    if knn:
        return (2, c.shape[0])
    if distance_metric.lower() == "precomputed":
        D = c
    else:
        D = distance.pdist(c, metric=distance_metric)
    return D.min(), D.max()


def delabel(ax):
    if isinstance(ax, numpy.ndarray):
        shape = ax.shape
        return numpy.asarray([delabel(ax_) for ax_ in ax]).reshape(shape)
    ax.set_xticks([])
    ax.set_xticklabels([])
    ax.set_yticks([])
    ax.set_yticklabels([])
    return ax


def first_stay(a, return_value=False):
    for i, e in enumerate(a[:-1]):
        if numpy.allclose(a[i + 1], e):
            if return_value:
                return i, e
            return i
    return None


def post_hoc_label_order(labels, data):
    sorter = pandas.DataFrame(dict(labels=labels, data=data))
    ordering = sorter.groupby("labels").data.mean().sort_values()
    out_labels = labels.copy()
    for i, ix in enumerate(ordering.index):
        out_labels[labels == ix] = i
    return out_labels
