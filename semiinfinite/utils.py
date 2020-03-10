import numpy as np

def numeric_gradient(f,x,h=1e-4,method='centered'):
    """h is the step size"""
    xtest = np.array(x)
    g = np.zeros(len(x))
    if method == 'centered':
        for i in xrange(len(x)):
            xtest[i] += h
            b = f(xtest)
            xtest[i] -= 2*h
            a = f(xtest)
            xtest[i] = x[i]

            g[i] = (b-a)
        g *= 1.0/(2*h)
        return g
    elif method == 'forward':
        a = f(x)
        for i in xrange(len(x)):
            xtest[i] += h
            b = f(xtest)

            g[i] = (b-a)
        g *= 1.0/h
        return g
    elif method == 'backward':
        a = f(x)
        for i in xrange(len(x)):
            xtest[i] -= h
            b = f(xtest)

            g[i] = (a-b)
        g *= 1.0/h
        return g
    else:
        raise ValueError("Invalid method %s"%(method,))

def test_gradient(f,fgrad,x,h=1e-4,epsilon=1e-3,name=""):
    """Prints whether the gradient of f is calculated correctly via fgrad at x.
    Errors are printed out if the approximation error exceeds epsilon.
    """
    g_numeric = numeric_gradient(f,x,h)
    g_calc = fgrad(x)
    diff = g_numeric-g_calc
    valid = True
    for i,v in enumerate(diff):
        if abs(v) > epsilon:
            print "test_gradient %s: errorneous value %d: numeric %g vs function %g"%(name,i,g_numeric[i],g_calc[i])
            valid = False
    return valid
