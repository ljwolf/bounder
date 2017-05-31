import numpy as np
from warnings import warn as Warn
from libpysal.weights import W
import pandas as pd
from . import utils
import networkx as nx
from collections import namedtuple

move = namedtuple('move', ('focal_id', 'focal_label', 
                  'neighbor_id', 'neighbor_label', 
                  'new_objective', 'current_objective',
                  'new_burden', 'current_burden',
                  'valid', 'type'))


class CachingCaller(object):
    """
    An object that stores its results in a private _trace list if asked to
    remember things. If not, the _trace only ever has the last called value. 
    """
    def __init__(self, fn, key,
                 initial_value = -np.nan, remember=True, 
                 rtol=1e-5, atol=1e-7):
        self.key = key
        self.fn = fn
        self._trace = [initial_value]
        self._remember = remember
        self._rtol = rtol
        self._atol = atol

    def __call__(self, grouper, cache=True, **kw):
        result = grouper[self.key].apply(self.fn, **kw).values
        if not cache:
            return result
        elif self._remember:
            self._trace.append(result)
        else:
            self._trace[-1] = result
        return result

    @property
    def last_eval(self):
        return self._trace[-1]

    @classmethod
    def from_kw(cls, **kw):
        if len(kw) == 1:
            key = list(kw.keys())[0]
            fn = kw[key]
            return cls(key=key, fn=fn)
        else:
            return [cls(fn=val, key=key) for key,val in kw.items()]

    def __gt__(self, other):
        return self.last_eval > other
    
    def __lt__(self, other):
        return self.last_eval < other

    def __eq__(self, other):
        return np.allclose(self.last_eval, other)

class Objective(CachingCaller):
    def __init__(self, fn, key, remember = True, **kw):
        CachingCaller.__init__(self, fn, key, remember=remember, **kw)

class Constraint(CachingCaller):
    def __init__(self, remember=True, **kw):
        rtol = kw.pop('rtol', 1e-5)
        atol = kw.pop('atol', 1e-7)
        initial_value = kw.pop('initial_value', -np.nan)
        if len(kw) < 1:
            raise KeyError("Key not provided with constraint function")
        if len(kw) != 1:
            raise TypeError("Multiple constraints provided to wrong constructor!")
        key = list(kw.keys())[0]
        fn = list(kw.values())[0]
        CachingCaller.__init__(self, fn, key, remember=remember, rtol=rtol,
                               atol=atol, initial_value = initial_value)

    def above(self, value, **kw):
        self.threshold = -value
        self._orig_fn = self.fn
        def neg_fn(*args, **kwargs):
            return -1 * self._orig_fn(*args, **kwargs)
        neg_fn.__dict__ = self.fn.__dict__
        self.fn = neg_fn
        return self

    def below(self, value, **kw):
        self.threshold = value
        return self

    def slack(self, data=None):
        if data is None:
            return self.last_eval - self.threshold
        else:
            return self(data) - self.threshold

    def satisfied(self, val=None):
        if val is None:
            return np.all(self.last_eval <= 0)
        else:
            return np.all(self(val, cache=False) <= 0)

class Regionalizer(object):
    def __init__(self, objective, constraints=None, remember=True):
        object.__init__(self)
        if not isinstance(objective, Objective):
            try:
                if isinstance(objective, tuple):
                    objective = dict(objective)
                assert len(objective) == 1
                key = list(objective.keys())[0]
                val = list(objective.values())[0]
                objective = Objective(key=key, fn=val, remember=remember)
            except:
                raise TypeError('Objective is not an "Objective" object'
                                ' and auto-construction failed!')
        self.objective = objective
        if constraints is None:
            constraints = tuple()
        elif isinstance(constraints, Constraint):
            constraints = (constraints, )
        self.constraints = constraints
        self.remember = remember

    def fit(self, n_clusters, *args, **kwargs):
        raise NotImplementedError("This must be implemented by the lower class")

    def refine(self, *args, **kwargs):
        raise NotImplementedError("This must be implemented by the lower class")
    
    def initialize(self, n_clusters, data, alist=None, 
                  w=None, initializer=None,
                  affinity_matrix = None,
                  **kw):
        """
        Initialize a contiguous solution to the clustering problem

        Arguments
        ----------
        alist       :   pandas Dataframe containing at least
                        "focal", "neighbor", and "weight" columns
        initializer :   method to initialize regions. Should have some method 
                        initializer.fit(X), which clusters the observations in X
                        and, after fitting, a "labels_" attribute which assigns
                        each row of data to a given region.
        """
        self.add_data(data, index_col = kw.pop('index_col', None), 
                            validate = kw.pop('validate', None))
        self.set_connections(alist=alist, w=w, verify=kw.pop('verify', True))
        self._validate_alist() #no need to validate objective for initialization
        self.n_clusters = n_clusters
        if initializer is None:
            from sklearn.cluster import AgglomerativeClustering
            initializer = AgglomerativeClustering(n_clusters = n_clusters, 
                                                  connectivity = self.w.sparse.toarray(),
                                                  **kw)
        labels = initializer.fit(data
                                 if affinity_matrix is None 
                                 else affinity_matrix).labels_
        self.initial_labels_ = labels
        self._data['current_labels'] = labels 
        label_label = self.make_alist_data('current_labels')
        self.alist['boundary'] = label_label['current_labels_focal'].values \
                                 != label_label['current_labels_neighbor'].values
        assert len(self._data.current_labels.unique()) == self.n_clusters
    
    def _update_boundary(self, data=None, **kw):
        if data is None:
            data = self._data
        label_label = self.make_alist_data('current_labels')
        self.alist['boundary'] = label_label['current_labels_focal'].values \
                                 != label_label['current_labels_neighbor'].values
    def add_data(self, data, 
                 index_col= None, validate=False):
        """
        Add data to the regionalizer.
        """
        if isinstance(data, np.ndarray):
            data = pd.DataFrame(data)
            data.columns = [''.join(('X', str(i))) for i in data.columns]
        if index_col is not None:
            data.index = data[index_col]
        if hasattr(self, '_data'):
            self._data = pd.concat((self._data, data), axis=1)
        else:
            self._data = data.copy(deep=True)
        if validate:
            jshape = pd.merge(self.alist, self._data, how='left', 
                              left_on='focal', right_index=True)\
                       .merge(self._data, how='left', left_on='neighbor', 
                              right_index=True, suffixes=('_focal', '_neighbor')).shape
            assert jshape[0] == self.alist.shape[0]
            assert set(self._data.index.tolist()) == (set(self.alist.focal.tolist()) 
                                                      | set(self.alist.neighbor.tolist()))

    def set_connections(self, alist=None, w=None, nx=None, verify=True):
        """
        Setup the graph structure. Will be kept under the hood as 
        an adjacency list and a PySAL W object, so if a networkx object
        is supplied, conversion is necessary.

        Only one argument is required to populate the connections. 
        """
        if all([nx is None, alist is None, W is None]):
            raise ValueError("At least one adjacency list, W, or networkx graph"
                             " must be provided to conduct clustering.")
        if nx is not None and all([alist is None, W is None]):
            w = W.from_networkx(nx)
        w, alist = utils.get_w_and_alist(alist=alist, W=w, 
                                           skip_verify=not verify)
        self.nx = nx if nx is not None else w.to_networkx()
        self.w = w
        self.alist = alist

    def make_alist_data(self, *columns):
        if len(columns) == 0:
            columns = self._data.columns.tolist()
        result = pd.merge(self.alist, self._data, how='left', 
                          left_on='focal', right_index=True)\
                   .merge(self._data, how='left', left_on='neighbor', 
                          right_index=True, suffixes=('_focal', '_neighbor'))
        if result.columns.duplicated().any():
            result = result.loc[:,~result.columns.duplicated()]
        return result

    def score(self, data=None, reduction=np.sum, **kw):
        """
        Score the regions by reducing a vector of region
         summaries to a single number. By default, this reduction is a sum. 
        """
        if data is None:
            data = self._data
        grouper = data.groupby('current_labels')
        return reduction(self.objective(grouper, **kw))
    
    def feasible(self, data=None, **kw):
        """
        Determine if the solution is feasible, meaning that the solution
        satisfies all constraints. 

        Arguments
        ----------
        data        :   the dataset over which to apply the constraints. 
                        if not provided, this defaults to the one attached
                        to the sampler. 
        kw          :   keyword argument dictionary to pass to the constraints
        """
        if data is None:
            data = self._data
        grouper = data.groupby('current_labels')
        result = all([constraint.satisfied(grouper, **kw)
                      for constraint in self.constraints])
        return result

    def connected(self, data=None, **kw):
        """
        Determine if the solution is connected, meaning that each
        region is contiguous
        """
        if data is None:
            data = self._data
        grouper = data.groupby('current_labels')
        def cxn_or_null(x):
            try:    
                return nx.is_connected(self.nx.subgraph(x.index))
            except nx.NetworkXPointlessConcept:
                return True
        connected = grouper.geometry.apply(cxn_or_null)
        self._was_connected = all(connected.values)
        return self._was_connected

    def valid(self, data=None, **kw):
        """
        Determine if the solution is valid, meaning that the solution is
        contiguous and feasible. 
        """
        return self.connected() and self.feasible(data=data, **kw)

    def slack(self, data=None):
        """
        Compute the slacks for the current solution or a new solution
        if provided.
        """
        if data is None:
            data = self._data
        grouper = data.groupby('current_labels')
        return np.vstack([constraint.slack(grouper) for constraint in self.constraints])

    def burden(self, data=None, net=True):
        """
        Compute the slacks for the current solution or a new solution 
        if provided. Then, assess whether the solution has 
        
        if net, will provide the sum of all burdens in the burden vector. 

        """
        if data is None:
            data = self._data
        slack = np.vstack(self.slack(data))
        burden = slack * (slack > 0).astype(int)
        assert np.all(burden >= 0)
        burden = burden.sum(axis=1)
        return np.sum(burden) if net else np.array(burden)

    def plot(self, **plot_kw):
        import matplotlib.pyplot as plt
        if 'linewidth' not in plot_kw:
            plot_kw['linewidth'] = 0
        ax = self._data.plot('current_labels', **plot_kw)
        ax.axis('off')
        return plt.gcf(), ax

    def plot_boundary(self, **plot_kw):
        import matplotlib.pyplot as plt
        if 'linewidth' not in plot_kw:
            plot_kw['linewidth'] = 0
        boundary_ixs = self.alist.query('boundary == True').focal
        bounds = self._data[self._data.index.isin(boundary_ixs)]
        ax = bounds.plot('current_labels', **plot_kw)
        ax.axis('off')
        return plt.gcf(), ax

    def _validate_keys(self):
        assert self.objective.key in self._data.columns
        for k in self.constraints.keys():
            assert k in self._data.columns

    def _validate_alist(self):
        np.testing.assert_array_equal(self.w.sparse.toarray(), 
                                   W.from_adjlist(self.alist).sparse.toarray())

    @property
    def grouper(self):
        return self._data.groupby('current_labels')

    @property
    def boundary_alist(self):
        self._update_boundary()
        return self.alist.query('boundary == True')

    def region_neighbors(self, target=None):
        """
        get a series keyed on `current_labels` with values containing arrays of the neighboring labels.
        """
        rneighbors = self.make_alist_data('current_labels')\
                         .query('boundary == True')\
                         .groupby('current_labels_focal')\
                         .current_labels_neighbor\
                         .unique()
        if target is None:
            return rneighbors
        else:
            return rneighbors.ix[target]

    def population(self, target=None):
        """
        Get a series eeyed on `current_labels` with values containing the number of atoms in each region
        """
        arbitrary_key = self._data.columns[0]
        counts = self._data.groupby('current_labels')[arbitrary_key].count()
        if target is None:
            return counts 
        else:
            return counts.ix[target]

    @property
    def labels(self):
        return self._data.current_labels.unique().tolist()


class Mogrifyer(object):
    """
    A class containing functions useful for split/merge operations on regions
    """
    def split(self, target_label=None):
        if target_region is None:
            target_region = self._identify_best_split()
    def merge(self, *labels, target_label=None):
        if isinstance(labels[0], move):
            target_label = labels[0].neighbor_label
            labels = [labels[0].focal_label]
        elif (len(labels) < 1 and target_label is not None) or (len(labels) < 2):
            raise ValueError("Number of labels supplied must be at least two")

        if target_label is None:
            target_label = labels[0] 
        labels = (label for label in labels if label != target_label)
        #print("merging {} into {}".format(labels, target_label))
        for label in labels:
            in_target = self._data.query('current_labels == {}'.format(label)).index
            self._data.ix[in_target, 'current_labels'] = target_label
            if self.remember:
                this_move = move(None, target_label, None, label, None, None, None, None, self.valid(), 'merge')
                self._trace.append(this_move)

    def find_best_merge(self, target_label):
        if target_label not in self._data.current_labels.unique():
            raise KeyError("provided target label ({}) not found in label set".format(target_label))
        candidates = self.region_neighbors(target_label)
        best_merge = move(None, None, None, None, -np.inf, -np.inf, np.inf, np.inf, False, 'initial')
        for candidate in candidates:
            local = self._data.copy(deep=True)
            current_obj = self.score(local)
            current_burden = self.burden(local, net=True)
            current_feasible = self.feasible(local)
            current_valid = self.valid(local)
            in_target = local.query('current_labels == {}'.format(target_label)).index
            print('considering merging {} into {}'.format(candidate, target_label))
            local.ix[in_target, 'current_labels'] = candidate
            merged_obj = self.score(local)
            merged_burden = self.burden(local)
            merged_feasible = self.feasible(local)
            merged_valid = self.valid(local)
            if (not current_feasible) and merged_feasible and (not best_merge.valid):
                best_merge = move(None, target_label, None, candidate, 
                                  merged_obj, current_obj, merged_burden, 
                                  current_burden, merged_feasible, 'merge-to-feasible')
            elif (not current_feasible) and merged_feasible and best_merge.valid:
                if merged_burden < best_merge.new_burden:
                    best_merge = move(None, target_label, None, candidate, 
                                      merged_obj, current_obj, merged_burden, 
                                      current_burden, merged_feasible, 'merge-unburden-to-feasible')
            elif (not current_feasible) and (not merged_feasible) and (not best_merge.valid):
                if (merged_burden < current_burden) and (merged_burden < best_merge.new_burden):
                    best_merge = move(None, target_label, None, candidate, 
                                      merged_obj, current_obj, merged_burden, 
                                      current_burden, merged_feasible, 'merge-unburden-infeasible')
            elif current_feasible and merged_feasible and best_merge.valid:
                if (merged_burden < current_burden) and (merged_burden < best_merge.new_burden):
                    best_merge = move(None, target_label, None, candidate, 
                                      merged_obj, current_obj, merged_burden, 
                                      current_burden, merged_feasible, 'merge-unburden')
        print('accepted merging {} into {}'.format(best_merge.neighbor_label, best_merge.focal_label))
        #if best_merge.type != 'initial':
        #    in_target = self._data.query('current_labels == {}'.format(best_merge.focal_label)).index
        #    self._data.ix[in_target, 'current_labels'] = best_merge.neighbor_label
        #if self.remember:
        #    self._trace.append(best_merge)
        return best_merge
