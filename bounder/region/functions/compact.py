import numpy as np

class Applicative(object):
    def __init__(self, fn):
        self.fn = fn
    @staticmethod
    def fn(*args, **kwargs):
        return fn(*args, **kwargs)
    def __call__(self, obj, **kwargs):
        return obj.apply(self.fn)

def isoperimetric_quotient(geom):
    return 4 * np.pi * geom.area * geom.boundary.length**-2
isoperimetric_quotient = Applicative(isoperimetric_quotient)

def perimeter_area_ratio(geom):
    return geom.area / geom.boundary.length
perimeter_area_ratio = Applicative(perimeter_area_ratio)

def boundary_amplitude(geom):
    return geom.convex_hull.boundary.length / geom.boundary.length
boundary_amplitude = Applicative(boundary_amplitude)

def convex_hull_ratio(geom):
    return geom.area / geom.convex_hull.area
convex_hull_ratio = Applicative(convex_hull_ratio)

def isoareal_quotient(geom):
    r = geom.boundary.length/(2 * np.pi)
    return np.pi * r**2 / geom.area 
isoareal_quotient = Applicative(isoareal_quotient)
