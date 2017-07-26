import numpy as np

class CachingCaller(object):
    """
    An object that stores its results in a private _trace list if asked to
    remember things. If not, the _trace only ever has the last called value. 
    """
    def __init__(self, fn, key,
                 initial_value = np.inf, remember=True, 
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
    def __init__(self, fn, key, remember = True, reduction=np.sum, **kw):
        CachingCaller.__init__(self, fn, key, remember=remember, **kw)
        self.reduction = reduction

    def __call__(self, grouper, cache=True, **kw):
        out = CachingCaller.__call__(self, grouper, cache=cache, **kw)
        return self.reduction(out)

class MultiObjective(Objective):
    def __init__(self, obj_components, 
                 outer_reduction=np.sum, 
                 inner_reductions=np.sum, 
                 weights=None, remember=True):
        self._remember = remember
        initial_value = np.inf
        self._trace = [initial_value]
        self._function_cache = list(map(lambda x: Objective(*x[::-1], #reverse, because function is first
                                                            remember=remember, 
                                                            reduction=outer_reduction), 
                                        obj_components.items()))
        if weights is None:
            weights = np.ones((len(obj_components))) / len(obj_components)
        self.weights = weights
        self.reduction = outer_reduction
    
    @classmethod
    def from_kw(cls, outer_reduction=np.sum, inner_reductions=np.sum, weights=None, remember=True, **fn_keys):
        return cls(obj_components=fn_keys, outer_reduction=outer_reduction, 
                   inner_reductions=inner_reductions, weights=weights, remember=remember)

    def __call__(self, grouper, cache=True, **kw):
        components = [fn(grouper) for fn in self._function_cache]
        result = self.reduction([comp*weight for comp,weight in zip(components, self.weights)])
        if not cache:
            return result
        elif self._remember:
            self._trace.append(result)
        else:
            self._trace[-1] = result
        return result

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
