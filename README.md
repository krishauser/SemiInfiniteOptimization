# SemiInfiniteOptimization

Code accompanying the paper "Semi-Infinite Programming for Trajectory Optimization with Nonconvex Obstacles" by K. Hauser

Kris Hauser
10/30/2018
kris.hauser@duke.edu

Underlying Python code is in the folder semiinfinite.   semiinfinite.sip is the core SIP code.


## Dependencies

semiinfinite.sip only depends on Numpy/Scipy.

The rest of the code requires the Klampt Python API (https://klampt.org) to be installed.  You will need to switch to the
logging_devel branches of both Klampt and Klampt/Cpp/Dependencies/KrisLibrary

The files in data/*.xml assume the Klampt folder is two levels up from this folder.

## Running demos

Basic geometry - geometry collision testing:

> python geomopt.py data/cube.off

Robot pose optimization testing:

> python robotposeopt.py data/tx90_geom_test.off

> python robotposeopt.py data/tx90_geom_test2.off

> python robotposeopt.py data/tx90_geom_test3.off

Trajectory optimization testing

> python robottrajopt.py data/tx90_geom_test3.off
