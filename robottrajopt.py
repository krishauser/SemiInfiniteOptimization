from klampt import *
from klampt.math import vectorops,so3,se3
from klampt.model.trajectory import SE3Trajectory,RobotTrajectory
from klampt.io import resource
from klampt import vis
import numpy as np
import sys
import time
from semiinfinite import geometryopt
from collections import defaultdict

gridres = 0.02
pcres = 0.02
DUMP_SDF = False
NUM_TRAJECTORY_DIVISIONS = 10
MORE_SUBDIVISION = 6
EDIT_ENDPOINTS = True
EDIT_INITIAL_GUESS = True
EDIT_OPTIMIZED_TRAJECTORY = False
SKIP_OPTIMIZATION = False
DEBUG_CONSTRAINTS = False
DRAW_GRID_AND_PC = False
EDIT_OBJECT_POSE = False
SHOW_TRAJECTORY_TRACE = False
PLOT_CSV_DISTANCES = True

#gridres = 0.1
#pcres = 0.05

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
#raw_input("Press enter to continue... ")

vis.add("world",world)
q0 = robot.getConfig()

kinematicscache = geometryopt.RobotKinematicsCache(robot,gridres,pcres)
trajcache = geometryopt.RobotTrajectoryCache(kinematicscache,NUM_TRAJECTORY_DIVISIONS*MORE_SUBDIVISION+MORE_SUBDIVISION-1)

if DUMP_SDF:
    for i in xrange(robot.numLinks()):
        fn = 'output/'+ robot.link(i).getName()+'.mat'
        print "Saving SDF to",fn
        geometryopt.dump_grid_mat(trajcache.kinematics.geometry[i].grid,fn)

try:
    trajinit = resource.get('robottrajopt_initial.path',default=None,doedit=False)
except Exception:
    trajinit = None
if trajinit is None:
    trajcache.qstart = q0[:]
    trajcache.qstart[1] = 1.0
    trajcache.qstart[2] = 1.8
    trajcache.qend = q0[:]
    trajcache.qend[1] = -1.0
    trajcache.qend[2] = 1.8
    trajcache.qstart = [0.0, 0.2399999999999998, 0.03999999999999922, -0.8000000000000002, -0.7400000000000002, -1.6000000000000005, 0.0]
    trajcache.qend = [0.0, -1.3000000000000003, 1.6199999999999997, 0.6999999999999997, 2.5200000000000014, -1.6000000000000008, 0.0]
    if EDIT_ENDPOINTS:
        ok,res = resource.edit("Trajectory endpoints",[trajcache.qstart,trajcache.qend],world=world)
        if ok:
            trajcache.qstart,trajcache.qend = res
            print trajcache.qstart
            print trajcache.qend
    xtraj = trajcache.straightLineState()
    trajinit = trajcache.stateToTrajectory(xtraj)
else:
    trajinit = RobotTrajectory(robot,trajinit.times,trajinit.milestones)
    if MORE_SUBDIVISION > 1:
        times = []
        for i in range(len(trajinit.times)-1):
            t0 = trajinit.times[i]
            t1 = trajinit.times[i+1]
            for k in range(MORE_SUBDIVISION):
                times.append(t0 + float(k)/float(MORE_SUBDIVISION)*(t1-t0))
        times.append(trajinit.times[-1])
        print "NEW TIMES",times
        trajinit = trajinit.remesh(times)[0]
    print "Trajectory has",len(trajinit.milestones),"milestones"    
    trajcache.qstart = trajinit.milestones[0][:]
    trajcache.qend = trajinit.milestones[-1][:]
    print "Time range",trajinit.times[0],trajinit.times[-1]
    trajcache.tstart = trajinit.times[0]
    trajcache.tend = trajinit.times[-1]
    xtraj = trajcache.trajectoryToState(trajinit)
    

if EDIT_INITIAL_GUESS:
    ok,res = resource.edit("Initial trajectory",trajinit,world=world)
    if ok:
        trajinit = res
        trajinit.times[-1] = trajcache.tend
        print "Saving to robottrajopt_initial.path"
        resource.set('robottrajopt_initial.path',res)

obstaclegeoms = [geometryopt.PenetrationDepthGeometry(obs.geometry(),None,pcres) for obs in obstacles]

ctest2 = geometryopt.RobotTrajectoryCollisionConstraint(obstaclegeoms,trajcache)

if DEBUG_CONSTRAINTS:
    print "Testing link 6 trajectory collision constraint"
    ctest = geometryopt.RobotLinkTrajectoryCollisionConstraint(robot.link(6),obstaclegeoms[0],trajcache)
    ctest.setx(xtraj)
    res = ctest.minvalue(xtraj)
    print "(Minimum distance, minimum parameter)",res
    ctest.clearx()
    print "Testing link 4 trajectory collision constraint"
    ctest = geometryopt.RobotLinkTrajectoryCollisionConstraint(robot.link(4),obstaclegeoms[0],trajcache)
    ctest.setx(xtraj)
    res = ctest.minvalue(xtraj)
    print "(Minimum distance, minimum parameter)",res
    ctest.clearx()
    #raw_input("Press enter to continue... ")
    print "Testing whole-robot trajectory collision constraint"

    ctest2.setx(xtraj)
    res = ctest2.minvalue(xtraj)
    print "(Minimum distance, minimum parameter)",res
    ctest2.clearx()
    raw_input("Press enter to continue... ")

def play_with_trajectory(traj,configs=[3]):
    vis.add("trajectory",traj)
    names = []
    for i,x in enumerate(traj.milestones):
        if i in configs:
            print "Editing",i
            names.append("milestone "+str(i))
            vis.add(names[-1],x[:])
            vis.edit(names[-1])
    #vis.addPlot("distance")
    vis.show()
    while vis.shown():
        vis.lock()
        t0 = time.time()
        updated = False
        for name in names:
            index = int(name.split()[1])
            qi = vis.getItemConfig(name)
            if qi != traj.milestones[index]:
                traj.milestones[index] = qi
                updated = True
        if updated:
            vis.add("trajectory",traj)
            xnew = trajcache.trajectoryToState(traj)
            ctest2.setx(xnew)
            res = ctest2.minvalue(xtraj)
            print res
            ctest2.clearx()
        vis.unlock()
        t1 = time.time()
        #vis.logPlot("timing","opt",t1-t0)
        time.sleep(max(0.001,0.025-(t1-t0)))

#play_with_trajectory(trajcache.stateToTrajectory(xtraj))

ctest2.verbose = 0
if SKIP_OPTIMIZATION:
    trajsolved = trajinit
    trace = [trajinit]
    params = [[]]
else:
    trajsolved,trace,params = geometryopt.optimizeCollFreeTrajectory(trajcache,trajinit,env=obstaclegeoms,constraints=[ctest2],greedyStart=False,verbose=1)
    #play_with_trajectory(trajsolved,[3])
    resource.set("robottrajopt_solved.path",trajsolved)

if PLOT_CSV_DISTANCES:
    fn = 'output/robottrajopt_distances.csv'
    print "Dumping pairwise distances to",fn,"..."
    f = open(fn,'w')
    f.write('t')
    for obs in obstacles:
        for i in xrange(robot.numLinks()):
            f.write(',d(%s;%s) init'%(robot.link(i).getName(),obs.getName()))
    for obs in obstacles:
        for i in xrange(robot.numLinks()):
            f.write(',d(%s;%s) opt'%(robot.link(i).getName(),obs.getName()))
    f.write('\n')
    ts = trajinit.times[0]
    te = trajinit.times[-1]
    numdivs = 200
    for d in xrange(numdivs+1):
        u = float(d)/numdivs
        t = ts + u*(te-ts)
        f.write("%g"%(t,))

        q = trajinit.eval(t)
        trajcache.kinematics.set(q)
        for j,obs in enumerate(obstacles):
            ds = [obstaclegeoms[j].distance(trajcache.kinematics.geometry[i])[0] for i in xrange(robot.numLinks())]
            for i in xrange(robot.numLinks()):
                f.write(',%g'%(ds[i],))
        trajcache.kinematics.clear()

        q = trajsolved.eval(t)
        trajcache.kinematics.set(q)
        for j,obs in enumerate(obstacles):
            ds = [obstaclegeoms[j].distance(trajcache.kinematics.geometry[i])[0] for i in xrange(robot.numLinks())]
            for i in xrange(robot.numLinks()):
                f.write(',%g'%(ds[i],))
        trajcache.kinematics.clear()
        f.write('\n')
    f.close()



if SHOW_TRAJECTORY_TRACE:
    for i,t in enumerate(trace):
        resource.edit("Trajectory %d/%d"%(i,len(trace)),t,"Trajectory",world=world)

xsolved = trajcache.trajectoryToState(trajsolved)
cps = defaultdict(list)
for p in params[0]:
    cps[(p[0],p[1])].append(p[2:])
print
print "Constrained points:"
ctest2.setx(xsolved)
for k in cps:
    cps[k] = sorted(cps[k])
    print "  link %d env %d: "%(k[0],k[1])
    for v in cps[k]:
        print "     ",v,"dist",ctest2.value(xsolved,k + v)
ctest2.clearx()
#raw_input("Press enter to continue > ")

""" #more debugging
print
ctest2.verbose = 2
ctest2.eval_minimum(xsolved)
raw_input("Press enter to continue > ")
"""


if EDIT_OPTIMIZED_TRAJECTORY:
    resource.edit("Solved trajectory",trajsolved,"Trajectory",world=world)
"""
xtraj = trajcache.trajectoryToState(trajsolved)
ctest2.setx(xtraj)
print "Resulting constraint residual",ctest2.minvalue(xtraj)
ctest2.clearx()
raw_input("Press enter to continue... ")
"""

timescale = 10
trajinit.times = [timescale*v for v in trajinit.times]
if trajinit is not trajsolved:
    trajsolved.times = [timescale*v for v in trajsolved.times]
eepos = [0,0,0.3]
vis.add("Initial trajectory",trajinit.getLinkTrajectory(6,0.1).getPositionTrajectory(eepos))
#vis.add("Initial trajectory",trajinit)
vis.setColor("Initial trajectory",1,1,0,0.5)
vis.hideLabel("Initial trajectory")
vis.setAttribute("Initial trajectory","width",2)
eetrajopt = trajsolved.getLinkTrajectory(6,0.1).getPositionTrajectory(eepos)
vis.add("Solution trajectory",eetrajopt)
#vis.add("Solution trajectory",trajsolved)
vis.hideLabel("Solution trajectory")
vis.animate(("world",robot.getName()),trajsolved)

params[0] = sorted(params[0],key=lambda x:x[2])
for i,p in enumerate(params[0]):
    link = p[0]
    env = p[1]
    t = p[2]
    wpt = p[3:6]
    eept = eetrajopt.eval(t*timescale)
    vis.add("support"+str(i),wpt)
    vis.add("tsupport"+str(i),eept)
    r = float(i)/len(params[0])
    if i%2 == 0:
        b = 0
    else:
        b = 0.5
    vis.setColor("support"+str(i),r,0,b)
    vis.setColor("tsupport"+str(i),r,0,b)
    vis.hideLabel("support"+str(i))
    vis.hideLabel("tsupport"+str(i))

qinit = q0

if DRAW_GRID_AND_PC:
    for i,g in enumerate(constraints[0].robot.geometry):
        if g is not None:
            vis.add("grid "+str(i),g.grid)
            vis.setColor("grid "+str(i),0,1,0,0.5)
            vis.hideLabel("grid "+str(i))
            #vis.add("pc "+str(i),g.pc)
            #vis.setColor("pc "+str(i),1,1,0,0.5)
    for i,obs in enumerate(ctest2.envs):
        vis.add("envpc "+str(i),obs.pc)
        vis.setColor("envpc "+str(i),1,1,0,0.5)
        vis.hideLabel("envpc "+str(i))

movableObjects = []
if EDIT_OBJECT_POSE:
    for i in xrange(world.numRigidObjects()):
        vis.edit(("world",world.rigidObject(i).getName()))
        movableObjects.append(("world",world.rigidObject(i).getName()))

#vis.addPlot("distance")

vis.show()
oldcps = []
while vis.shown():
    vis.lock()
    t0 = time.time()
    if EDIT_OBJECT_POSE:
        for path in movableObjects:
            q = vis.getItemConfig(path)
            T = (q[:9],q[9:])
            for c,p in zip(constraints,pairs):
                if p[1].getName() == path[1]:
                    c.env.setTransform(T)
        distances = [c.eval_minimum(qcollfree) for c in constraints]
        print "Distances",distances

    for i in oldcps:
        vis.hide(i)
    oldcps = []
    for i in xrange(len(cps)):
        for j in xrange(len(cps[i])):
            name = "cp(%d,%d)"%(i,j)
            vis.add(name,cps[i][j])
            vis.setColor(name,0,0,0,1)
            vis.hideLabel(name)
            oldcps.append(name)
    vis.unlock()
    t1 = time.time()
    #vis.logPlot("timing","opt",t1-t0)
    time.sleep(max(0.001,0.05-(t1-t0)))
vis.kill()