from klampt import *
from klampt.math import vectorops,so3,se3
from klampt.model.trajectory import SE3Trajectory
from klampt.io import resource
from klampt import vis
import numpy as np
import sys
import time
from semiinfinite import geometryopt
from semiinfinite.sip import SemiInfiniteOptimizationSettings

gridres = 0.02
pcres = 0.02

#gridres = 0.1
#pcres = 0.05

geometryopt.TEST_PYCCD = False

DUMP_SDF = False
DRAW_GRID_AND_PC = False
VERBOSE = 0

world = WorldModel()

if len(sys.argv) > 1:
    for fn in sys.argv[1:]:
        world.readFile(fn)
else:
    res = resource.load('WorldModel')
    if res is not None:
        fn,world = res
    else:
        exit(1)

if world.numRobots() == 0:
    print "Must specify a robot"
    exit(1)

robot = world.robot(0)
obstacles = []
for i in xrange(1,world.numRobots()):
    for j in xrange(world.robot(i).numLinks()):
        obstacles.append(world.robot(i).link(j))
for i in xrange(world.numRigidObjects()):
    obstacles.append(world.rigidObject(i))
#for i in xrange(world.numTerrains()):
#   obstacles.append(world.terrain(i))
print "%d robots, %d rigid objects, %d terrains"%(world.numRobots(),world.numRigidObjects(),world.numTerrains())
assert len(obstacles) > 0
constraints,pairs = geometryopt.makeCollisionConstraints(robot,obstacles,gridres,pcres)
print "Created",len(constraints),"constraints"

vis.add("world",world)
movableObjects = []
for i in xrange(world.numRigidObjects()):
    vis.edit(("world",world.rigidObject(i).getName()))
    movableObjects.append(("world",world.rigidObject(i).getName()))

#extract geometries from constraints
linkgeoms = [None]*robot.numLinks()
obstaclegeoms = [None]*len(obstacles)
for c,(link,obj) in zip(constraints,pairs):
    for i,obs in enumerate(obstacles):
        if obj is obs:
            obstaclegeoms[i] = c.env
    linkgeoms[link.index] = c.robot.geometry[link.index]
assert all(o is not None for o in linkgeoms),"Hm... couldn't find link geometries?"
assert all(o is not None for o in obstaclegeoms),"Hm... couldn't find obstacle geometries?"
#this recreates the geometries too much
#obstaclegeoms = [PenetrationDepthGeometry(obs.geometry(),gridres,pcres) for obs in obstacles]

if DUMP_SDF:
    for i in xrange(robot.numLinks()):
        fn = 'output/'+ robot.link(i).getName()+'.mat'
        print "Saving SDF to",fn
        geometryopt.dump_grid_mat(linkgeoms[i].grid,fn)

qinit = robot.getConfig()
vis.add("qsoln",qinit)
vis.setColor("qsoln",0,1,0,0.5)
#edit configuration as target
vis.edit(("world",robot.getName()))

#should we draw?
if DRAW_GRID_AND_PC:
    for i,g in enumerate(constraints[0].robot.geometry):
        if g is not None:
            vis.add("grid "+str(i),g.grid)
            vis.setColor("grid "+str(i),0,1,0,0.5)
            vis.hideLabel("grid "+str(i))
            #vis.add("pc "+str(i),g.pc)
            #vis.setColor("pc "+str(i),1,1,0,0.5)
    for i,obs in enumerate(obstaclegeoms):
        vis.add("envpc "+str(i),obs.pc)
        vis.setColor("envpc "+str(i),1,1,0,0.5)
        vis.hideLabel("envpc "+str(i))

settings = None
settings = SemiInfiniteOptimizationSettings()
#if you use qinit = random-collision-free, you'll want to set this higher
settings.max_iters = 5
settings.minimum_constraint_value = -0.02

vis.addPlot("timing")

vis.show()
oldcps = []
while vis.shown():
    vis.lock()
    t0 = time.time()
    for path in movableObjects:
        q = vis.getItemConfig(path)
        T = (q[:9],q[9:])
        for c,p in zip(constraints,pairs):
            if p[1].getName() == path[1]:
                c.env.setTransform(T)
    q0 = robot.getConfig()
    robot.setConfig(qinit)
    #qcollfree,trace,cps = geometryopt.optimizeCollFreeRobot(robot,obstacles,constraints=constraints,qinit='random-collision-free',qdes=q0,verbose=VERBOSE,settings=settings)
    qcollfree,trace,cps = geometryopt.optimizeCollFreeRobot(robot,obstacles,constraints=constraints,qinit=None,qdes=q0,verbose=VERBOSE,settings=settings)

    #vis.add("transformTrace",trace)
    assert len(qcollfree) == robot.numLinks()
    #robot.setConfig(qcollfree)
    vis.setItemConfig("qsoln",qcollfree)

    if geometryopt.TEST_PYCCD:
        gx = [c(qcollfree) for c in constraints]
    else:
        gx = [c.eval_minimum(qcollfree) for c in constraints]
    feasible = all([v >= 0 for v in gx])
    if feasible:
        vis.setColor("qsoln",0,1,0,0.5)
    else:
        vis.setColor("qsoln",1,0,0,0.5)

    #initialize the next step from the last solved configuration
    qinit = qcollfree
    """
    #debug printing
    for c in constraints:
        c.setx(qcollfree)
    distances = [c.minvalue(qcollfree) for c in constraints]
    for c in constraints:
        c.clearx()
    print "Distances",distances
    """
    for i in oldcps:
        vis.hide(i)
    oldcps = []
    for i in xrange(len(cps)):
        for j in xrange(len(cps[i])):
            name = "cp(%d,%d)"%(i,j)
            vis.add(name,cps[i][j][:3])
            vis.setColor(name,0,0,0,1)
            vis.hideLabel(name)
            oldcps.append(name)
    #let the goal wander around
    #q0 = qcollfree
    #let the seed wander around
    #qinit = qcollfree
    vis.unlock()
    t1 = time.time()
    vis.logPlot("timing","opt",t1-t0)
    time.sleep(max(0.001,0.05-(t1-t0)))
vis.kill()