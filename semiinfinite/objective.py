import numpy as np
import scipy
import scipy.sparse

class ObjectiveFunctionInterface:
    """Base class for an objective function. 

    You can modify and combine objective functions by basic arithemetic and power
    operations.
    """
    
    def __init__(self):
        pass
    
    def value(self,x):
        """Returns the objective function value at x"""
        raise NotImplementedError()
    
    def minstep(self,x):
        """Returns a shift dx that locally minimizes f(x+dx) up to second order, i.e.
        dx = -H^-1*g, with H = self.hessian(x), g = self.gradient(x)
        """
        raise NotImplementedError()
    
    def gradient(self,x):
        """Compute the gradient of the objective f(x).  Not needed if minstep is implemented"""
        raise NotImplementedError()
    
    def hessian(self,x):
        """Compute / approximate the hessian of objective at x.  Letting H=hessian(),
        the QP optimizer will try to minimize ||*(dx-minstep()||_H^2 = (dx-minstep())^T H (dx-minstep()).

        Returns a scalar, 1D numpy array, 2D numpy array, or a 2D sparse array.

        A scalar return value is interpreted as a scaling of the identity matrix.

        A 1D array return value is interpreted as W=diag(hessian()).
        """
        return 1

    def integrate(self,x,dx):
        return np.asarray(x)+np.asarray(dx)

    def __add__(self,rhs):
        if isinstance(rhs,ObjectiveFunctionInterface):
            return AddObjectiveFunction(self,rhs)
        else:
            return AddObjectiveFunction(self,ConstantObjectiveFunction(rhs))
    def __sub__(self,rhs):
        if isinstance(rhs,ObjectiveFunctionInterface):
            return SubObjectiveFunction(self,rhs)
        else:
            return SubObjectiveFunction(self,ConstantObjectiveFunction(rhs))
    def __mul__(self,rhs):
        if isinstance(rhs,ObjectiveFunctionInterface):
            return MulObjectiveFunction(self,rhs)
        else:
            return MulConstObjectiveFunction(self)
    def __div__(self,rhs):
        if isinstance(rhs,ObjectiveFunctionInterface):
            return DivObjectiveFunction(self,rhs)
        else:
            return MulConstObjectiveFunction(self,1.0/rhs)
    def __pow__(self,rhs):
        if isinstance(rhs,ObjectiveFunctionInterface):
            raise NotImplementedError("Can't raise a function to a function power")
        else:
            return PowConstObjectiveFunction(self,rhs)
    def __radd__(self,rhs):
        return AddObjectiveFunction(ConstantObjectiveFunction(rhs),self)
    def __rsub__(self,rhs):
        return SubObjectiveFunction(ConstantObjectiveFunction(rhs),self)
    def __rmul__(self,rhs):
        return MulConstObjectiveFunction(self,rhs)
    def __rdiv__(self,rhs):
        return MulConstObjectiveFunction(self,1.0/rhs)


class ConstantObjectiveFunction(ObjectiveFunctionInterface):
    """A function f(x) = c."""
    def __init__(self,c):
        self.c = c
    def value(self,x):
        return self.c
    def minstep(self,x):
        return np.zeros(x.shape)
    def gradient(self,x):
        return np.zeros(x.shape)
    def hessian(self,x):
        return 0


class LinearObjectiveFunction(ObjectiveFunctionInterface):
    """A function f(x) = c^T x

    Note that the way that we're defining the QP as minimizing the
    deviation from a given step size, the step size is
    controlled by the length of the c vector.
    """
    def __init__(self,c):
        self.c = scipy.sparse.coo_matrix(c)
        self.hess = (self.c*self.c.T)/self.c.dot(self.c)
    def value(self,x):
        return self.c.dot(x)
    def minstep(self,x):
        return -self.c
    def gradient(self,x):
        return self.c
    def hessian(self,x):
        return self.hess


class AddObjectiveFunction(ObjectiveFunctionInterface):
    """A function f(x) = g(x)+h(x)."""
    def __init__(self,g,h):
        self.g = g
        self.h = h
    def value(self,x):
        return self.g.value(x)+self.h.value(x)
    def gradient(self,x):
        return self.g.gradient(x)+self.h.gradient(x)
    def hessian(self,x):
        return self.g.hessian(x)+self.h.hessian(x)
    def integrate(self,x,dx):
        return self.g.integrate(x,dx)


class SubObjectiveFunction(ObjectiveFunctionInterface):
    """A function f(x) = g(x)-h(x)."""
    def __init__(self,g,h):
        self.g = g
        self.h = h
    def value(self,x):
        return self.g.value(x)-self.h.value(x)
    def gradient(self,x):
        return self.g.gradient(x)-self.h.gradient(x)
    def hessian(self,x):
        return self.g.hessian(x)-self.h.hessian(x)
    def integrate(self,x,dx):
        return self.g.integrate(x,dx)


class MulConstObjectiveFunction(ObjectiveFunctionInterface):
    """A function f(x) = g(x)*h."""
    def __init__(self,g,h):
        self.g = g
        self.h = h
    def value(self,x):
        return self.g.value(x)*self.h
    def gradient(self,x):
        return self.g.gradient(x)*self.h
    def hessian(self,x):
        return self.g.hessian(x)*self.h
    def minstep(self,x):
        return self.g.minstep(x)
    def integrate(self,x,dx):
        return self.g.integrate(x,dx)


class MulObjectiveFunction(ObjectiveFunctionInterface):
    """A function f(x) = g(x)*h(x)."""
    def __init__(self,g,h):
        self.g = g
        self.h = h
    def value(self,x):
        return self.g.value(x)*self.h.value(x)
    def gradient(self,x):
        return self.g.gradient(x)*self.h.value(x)+self.g.value(x)*self.h.gradient(x)
    def hessian(self,x):
        return self.g.hessian(x)*self.h.value(x)+np.outer(self.g.gradient(x),self.h.gradient(x))+self.g.value(x)*self.h.hessian(x)+np.outer(self.h.gradient(x),self.g.gradient(x))
    def integrate(self,x,dx):
        return self.g.integrate(x,dx)


class DivObjectiveFunction(ObjectiveFunctionInterface):
    """A function f(x) = g(x)/h(x)."""
    def __init__(self,g,h):
        self.g = g
        self.h = h
    def value(self,x):
        return self.g.value(x)/self.h.value(x)
    def gradient(self,x):
        hval = self.h.value(x)
        return (self.g.gradient(x)*hval+self.g.value(x)*self.h.gradient(x))/(hval*hval)
    def hessian(self,x):
        raise NotImplementedError("Can't do hessian of function/function division yet")
    def integrate(self,x,dx):
        return self.g.integrate(x,dx)


class PowConstObjectiveFunction(ObjectiveFunctionInterface):
    """A function f(x) = g(x)**h."""
    def __init__(self,g,h):
        self.g = g
        self.h = h
    def value(self,x):
        return pow(self.g.value(x),self.h)
    def gradient(self,x):
        return self.h*pow(self.g.value(x),self.h-1)*self.g.gradient(x)
    def hessian(self,x):
        gval = self.g.value(x)
        ggrad = self.g.gradient(x)
        return self.h*(self.h-1)*pow(gval,self.h-2)*np.outer(ggrad,ggrad) + self.h*pow(gval,self.h-1)*self.g.hessian(x)
    def integrate(self,x,dx):
        return self.g.integrate(x,dx)
