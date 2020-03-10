import numpy as np


class ConstraintInterface:
    """A constraint is a function of the form f(x) >= 0 where x is in R^n.

    To allow for caching between multiple constraints, to retrieve f(x) the methods setx,
    value, and clearx are called in the format:
      f.setx(x)
      fx = f.value(x)
      f.clearx()
      #result is in fx
    """
    def __init__(self):
        pass
    def dims(self):
        """Returns the number of dimensions of f(x).  0 indicates a scalar."""
        return 0
    def __call__(self,x):
        """Evaluates f(x)"""
        self.setx(x)
        fx = self.value(x)
        self.clearx()
        return fx
    def setx(self,x):
        """Called with the x value before value or df_dx Can set a cache here."""
        pass
    def clearx(self):
        """Called after all calls to value or df_dx with a given x.  Can clear a cache here."""
        pass
    def value(self,x):
        """Returns the function value at the optimization variable x. x must be previously set using setx."""
        raise NotImplementedError()
    def df_dx(self,x):
        """Returns the Jacobian or gradient of the value with respect to x.
        x must be previously set using setx."""
        raise NotImplementedError()
    def df_dx_numeric(self,x,h=1e-4):
        """Helper function: evaluates df_dx using numeric differentiation"""
        self.clearx()
        res = numeric_gradient(lambda v:self.__call__(v),x,h)
        self.setx(x)
        return res


class DomainInterface:
    """A representation of some domain for use in SemiInfiniteConstraint.domain()"""
    def sample(self):
        raise NotImplementedError()
    def extrema(self):
        raise NotImplementedError()


class IntervalDomain(DomainInterface):
    """A 1D interval"""
    def __init__(self,vmin,vmax):
        self.interval = (vmin,vmax)
    def sample(self):
        return random.uniform(self.vmin,self.vmax)
    def extrema(self):
        return [self.vmin,self.vmax]


class SetDomain(DomainInterface):
    """A discrete set of items"""
    def __init__(self,items):
        self.items = list(items)
    def sample(self):
        return random.choice(self.items)
    def extrema(self):
        return self.items


class CartesianProductDomain(DomainInterface):
    """A cartesian product of multiple domains"""
    def __init__(self,domains):
        self.domains = domains
    def sample(self):
        return np.hstack([d.sample() for d in self.domains])
    def extrema(self):
        import itertools
        return [np.hstack(element) for element in itertools.product([d.extrema() for d in self.domains])]


class UnionDomain(DomainInterface):
    """A union of multiple domains"""
    def __init__(self,domains):
        self.domains = domains
    def sample(self):
        d = random.choice(self.domains)
        return d.sample()
    def extrema(self):
        return sum([d.extrema() for d in self.domains],[])
        

class SemiInfiniteConstraintInterface:
    """A semi-infinite constraint is of the form f(x,y) >= 0 for all y in Y, where
    x is in R^n and Y is a subset of R^m.

    To allow for caching between multiple constraints, to retrieve f(x,y) the methods setx,
    value, and clearx are called in the format:
      f.setx(x)
      fxy = f.value(x,y)
      f.clearx()
      #result is in fxy
    """
    def __init__(self):
        pass
    def dims(self):
        """Returns the number of dimensions of f(x).  0 indicates a scalar."""
        return 0
    def __call__(self,x,y):
        """Evaluates f(x,y).  No need to override this"""
        self.setx(x)
        res = self.value(x,y)
        self.clearx()
        return res
    def eval_minimum(self,x):
        """Helper: evaluates min_{y in Y} f(x,y) without caching."""
        self.setx(x)
        newpts = self.minvalue(x)
        self.clearx()
        if len(newpts) > 0 and not hasattr(newpts[0],'__iter__'): #it's just a plain pair
            return newpts[0]
        if len(newpts) > 0:
            return min(dmin for (dmin,param) in newpts)
        raise ValueError("minvalue does not appear to return a minimum value...")

    def setx(self,x):
        """Called with the x value before value, minvalue, oracle, df_dx, or df_dy.
        Can set a cache here."""
        pass
    def clearx(self):
        """Called after all calls to value, minvalue, oracle, df_dx, or df_dy with
        a given x.  Can clear a cache here."""
        pass
    def value(self,x,y):
        """Returns the function value at the optimization variable x and parameter y.
        x must be previously set using setx."""
        raise NotImplementedError()
    def minvalue(self,x,bound=None):
        """Returns a pair (fmin,param) containing (min_y f(x,y), arg min_y f(x,y)).
        In other words, fmin=f(x,param).

        The optimizer will add these values to the constraint list.

        x must be previously set using setx.

        Alternatively, can return a list of such pairs.  This may be useful when you can
        efficiently produce multiple local minima."""
        raise NotImplementedError()
    def df_dx(self,x,y):
        """Returns the Jacobian or gradient of the value with respect to x.
        x must be previously set using setx."""
        raise NotImplementedError()
    def df_dy(self,x,y):
        """Returns the Jacobian of the value with respect to param. Not currently used"""
        raise NotImplementedError()
    def domain(self):
        """Returns a representation of the domain.  Return value is an instance of
        DomainInterface."""
        raise NotImplementedError()
    def df_dx_numeric(self,x,y,h=1e-4):
        """Helper function: properly evaluates df_dx using numeric differentiation"""
        self.clearx()
        res = numeric_gradient(lambda v:self.__call__(v,y),x,h)
        self.setx(x)
        return res
    def df_dy_numeric(self,x,y,h=1e-4):
        """Helper function: properly evaluates df_dy using numeric differentiation"""
        return numeric_gradient(lambda v:self.value(x,v),y,h)


class MultiSemiInfiniteConstraint(SemiInfiniteConstraintInterface):
    """A semi-infinite constraint composed of multiple semi-infinite constraints
    c1,...,cn, all of which are scalars.  The minimum result is 

    min_y f(x,y) = min(min_y1 c1(x,y1),...,min yn cn(x,yn))

    The co-parameter domain is y = (i,yi,[0 padding]) where i is the index of
    the constraint, yi is the co-parameter of constraint i, and there is enough
    padding to make an even-sized co-parameter.

    Having a single MultiSemiInfiniteConstraint can be a lot faster than inputting
    many constraints into a semi-infinite optimization.  However, each of your
    constraints should handle the `bound` argument to `minvalue`.
    """
    def __init__(self,constraints,all_negative=True):
        """Initializes with a list of constraints.  If all_negative=True (default),
        then all constraints with negative values are generated on a minvalue call.
        """
        self.constraints = constraints
        self.all_negative = all_negative
        self.num_co_parameters = []
        for c in self.constraints:
            assert c.dims() == 0
            pexample = sampleDomain(c.domain())
            self.num_co_parameters.append(len(pexample))
        self.max_co_parameters = max(self.num_co_parameters)
    
    def __call__(self,x,y):
        index = int(y[0])
        assert index >= 0 and index < len(self.constraints)
        self.constraints[index].setx(x)
        res = self.value(x,y)
        self.constraints[index].clearx()
        return res

    def setx(self,x):
        for c in self.constraints:
            c.setx(x)
    def clearx(self):
        for c in self.constraints:
            c.clearx()
    def value(self,x,y):
        index = int(y[0])
        assert index >= 0 and index < len(self.constraints)
        return self.constraints[index].value(x,y[1:1+self.num_co_parameters[index]])
    def minvalue(self,x,bound=None):
        if self.all_negative:
            res = []
            for i,c in enumerate(self.constraints):
                if bound is not None and bound < 0:
                    bound = 0
                mv = c.minvalue(x,bound)
                if len(mv) > 0 and not hasattr(mv[0],'__iter__'): #it's just a plain pair
                    mv = [mv]
                for (fi,yi) in mv:
                    padding = self.max_co_parameters - len(yi)
                    y = np.hstack((i,yi,[0]*padding))
                    res.append((fi,y))
            return res
        #only return best
        best = None
        for i,c in enumerate(self.constraints):
            mv = c.minvalue(x,bound)
            if len(mv) > 0 and not hasattr(mv[0],'__iter__'): #it's just a plain pair
                mv = [mv]
            for (fi,yi) in mv:
                if bound is None or fi < bound:
                    bound = fi
                    padding = self.max_co_parameters - len(yi)
                    best = np.hstack((i,yi,[0]*padding))
        if best is None:
            return []
        return (bound,best)
    def df_dx(self,x,y):
        index = int(y[0])
        assert index >= 0 and index < len(self.constraints)
        return self.constraints[index].df_dx(x,y[1:1+self.num_co_parameters[index]])
    def df_dy(self,x,y):
        index = int(y[0])
        assert index >= 0 and index < len(self.constraints)
        return np.hstack(([0],self.constraints[index].df_dy(x,y[1:1+self.num_co_parameters[index]])))
    def domain(self):
        raise NotImplementedError("TODO: Can't inspect domain of a MultiSemiInfiniteConstraint yet")


class MinimumConstraintAdaptor(ConstraintInterface):
    """Given a semi-infinite constraint, returns the standard constraint f(x) = min_y f(x,y) >= 0. """
    def __init__(self,semi_infinite_constraint):
        self.f = semi_infinite_constraint
        self.minval = None
        self.minparam = None
    def dims(self):
        return self.f.dims()
    def setx(self,x):
        self.f.setx()
        res = self.minvalue(x)
        if len(res) > 0 and hasattr(res[0],'__iter__'):
            res = res[0]
        self.minval,self.minparam = res
    def clearx(self):
        self.f.clearx()
    def value(self,x):
        return self.minval
    def df_dx(self,x):
        return self.f.df_dy(x,self.minparam)


class SemiInfiniteConstraintAdaptor(SemiInfiniteConstraintInterface):
    """Turns a standard subclass of ConstraintInterface into a SemiInfiniteConstraintInterface
    with domain {0}
    """
    def __init__(self,constraint):
        self.constraint = constraint
    def dims(self):
        return self.constraint.dims()
    def setx(self,x):
        self.constraint.setx(x)
    def clearx(self):
        self.constraint.clearx()
    def value(self,x,y):
        return self.constraint.value(x)
    def minvalue(self,x,bound=None):
        return (self.constraint.value(x),0)
    def df_dx(self,x,y):
        return self.constraint.df_dx(x)
    def df_dy(self,x,y):
        d = self.dims()
        if dims == 0: return 0
        return np.zeros(d)
    def domain(self):
        return SetDomain([0])
    
