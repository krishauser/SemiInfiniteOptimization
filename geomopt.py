from klampt import *
from klampt.math import vectorops,so3,se3
from klampt.model.trajectory import Trajectory,SE3Trajectory
from klampt import vis
from klampt.io import resource
import numpy as np
import sys
import time
from semiinfinite.geometryopt import *

gridres = 0.05
pcres = 0.02

if len(sys.argv) > 1:
    fn = sys.argv[1]
    resource.setDirectory('.')
    obj = resource.get(fn,'Geometry3D')
else:
    res = resource.load('Geometry3D')
    if res is not None:
        fn,obj = res
    else:
        exit(1)


if obj is None:
    exit(1)

print "Input object has type",obj.type(),"with",obj.numElements(),"elements"
geom1 = PenetrationDepthGeometry(obj,gridres,pcres)
geom2 = PenetrationDepthGeometry(obj.clone(),gridres,pcres)
        
geom2.setTransform((so3.identity(),[1.2,0,0]))
vis.add("obj1",geom1.grid)
vis.add("obj2",geom2.pc)
vis.setColor("obj1",1,1,0,0.5)
vis.setColor("obj2",0,1,1,0.5)
vis.setAttribute("obj2","size",3)


vis.addPlot("timing")

vis.add("transform",se3.identity())
vis.edit("transform")
vis.show()
oldcps = []
while vis.shown():
    vis.lock()
    t0 = time.time()
    q = vis.getItemConfig("transform")
    T = (q[:9],q[9:])

    #debug optimization
    geom1.setTransform(T)
    Tcollfree,trace,cps = optimizeCollFree(geom1,geom2,T,verbose=0)
    traceTraj = SE3Trajectory(range(len(trace)),trace)
    vis.add("transformTrace",traceTraj)
    for i in oldcps:
        vis.hide(i)
    oldcps = []
    for i in xrange(len(cps)):
        name = "cp"+str(i+1)
        vis.add(name,cps[i])
        vis.setColor(name,0,0,0,1)
        oldcps.append(name)
    geom1.setTransform(Tcollfree)

    vis.unlock()
    t1 = time.time()
    vis.logPlot("timing","opt",t1-t0)
    time.sleep(max(0.001,0.05-(t1-t0)))
vis.kill()