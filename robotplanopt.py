from __future__ import print_function,division
from klampt import *
from klampt import vis
from klampt.io import resource,loader
import numpy as np
import sys
import time
from semiinfinite import geometryopt,planopt

#number of trials to be run for doAllTests
numAllTrials = 10
#number of restarts planned for restart/hybrid methods
numRestarts = 5
#number of optimization iterations to run per path for hybrid methods
optimizationMaxIters = 50
#total maximum time for each planning run
totalMaxTime = 30.0

#resolution of the signed distance fields used for the robot
gridres = 0.02
#resolution of the point cloud for the object
pcres = 0.02
#if true, dumps each of the robot's links SDFs to .mat files (useful for debugging, see SDFPlotting.ipynb)
DUMP_SDF = False


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
    print("Must specify a robot")
    exit(1)

robot = world.robot(0)
print("%d robots, %d rigid objects, %d terrains"%(world.numRobots(),world.numRigidObjects(),world.numTerrains()))
#raw_input("Press enter to continue... ")

vis.add("world",world)
q0 = robot.getConfig()
qtarget = robot.getConfig()
configs = [q0,qtarget]
save,result = resource.edit('start and goal',configs,'Configs',world=world)
configs = result
vis.add("start",configs[0])
vis.add("target",configs[1])
vis.setColor("start",0,1,0,0.5)
vis.setColor("target",1,0,0,0.5)

print("Initializing geometries for semi-infinite optimization")
kinematicscache = geometryopt.RobotKinematicsCache(robot,gridres,pcres)

if DUMP_SDF:
    for i in xrange(robot.numLinks()):
        fn = 'output/'+ robot.link(i).getName()+'.mat'
        print("Saving SDF to",fn)
        geometryopt.dump_grid_mat(kinematicscache.geometry[i].grid,fn)

def dotest(name,numTrials,**args):
    logFile = 'output/robotplanopt_'+name+'.csv'
    print("*** Dumping log to %s ***"%(logFile,))
    flog = open(logFile,'w')
    
    try:
        for trial in xrange(numTrials):
            robot.setConfig(configs[0])
            res = planopt.planOptimizedTrajectory(world,robot,configs[1],kinematicsCache=kinematicscache,logFile=flog,**args)
            if res is not None:
                loader.save(res,'auto','output/robotplanopt_'+name+"_"+str(trial)+".path")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print("Exception encountered during dotest",name,":",e)
        res = None

    print("*** Log dumped to %s ***"%(logFile,))
    flog.close()
    return res

#movingSubset = 'auto'
#otherwise only a few links will move
movingSubset = [1,2,3,4,5,6]
#movingSubset = [1,2,3]

#hack for joint limits
qmin,qmax = robot.getJointLimits()
q = robot.getConfig()
for i in range(robot.numLinks()):
    if i not in movingSubset:
        qmin[i] = qmax[i] = q[i]
robot.setJointLimits(qmin,qmax)


def doRRTStar(numTrials=1):
    plannerSettings={'type':'rrt*','optimizing':True,'movingSubset':movingSubset}
    return dotest('rrtstar',numTrials,plannerSettings=plannerSettings,numRestarts=1,plannerMaxIters=200000,plannerMaxTime=totalMaxTime,optimizationMaxIters=0)

def doLazyRRGStar(numTrials=1):
    plannerSettings={'type':'lazyrrg*','optimizing':True,'movingSubset':movingSubset}
    return dotest('lazyrrtstar',numTrials,plannerSettings=plannerSettings,numRestarts=1,plannerMaxIters=200000,plannerMaxTime=totalMaxTime,optimizationMaxIters=0)

def doRestartSBL(numTrials=1):
    plannerSettings={'type':'sbl','perturbationRadius':1.0,'bidirectional':True}
    return dotest('restart_sbl',numTrials,plannerSettings=plannerSettings,numRestarts=numRestarts*4,plannerMaxIters=200000,plannerMaxTime=totalMaxTime/(numRestarts*2),optimizationMaxIters=0)

def doRestartShortcutSBL(numTrials=1):
    plannerSettings={'type':'sbl','perturbationRadius':1.0,'bidirectional':True,'shortcut':True, 'optimizing':True,'movingSubset':movingSubset}
    return dotest('restart_shortcut_sbl',numTrials,plannerSettings=plannerSettings,numRestarts=numRestarts*2,plannerMaxIters=200000,plannerMaxTime=totalMaxTime/(numRestarts*2),optimizationMaxIters=0)

def doHybridRRTStar(numTrials=1):
    plannerSettings={'type':'rrt*','optimizing':True,'movingSubset':movingSubset}
    return dotest('hybrid_rrtstar',numTrials,plannerSettings=plannerSettings,maxTime=totalMaxTime,numRestarts=numRestarts,plannerContinueOnRestart=True,plannerMaxIters=200000,plannerMaxTime=totalMaxTime/(2*numRestarts),optimizationMaxIters=optimizationMaxIters*numRestarts)

def doHybridLazyRRGStar(numTrials=1):
    plannerSettings={'type':'lazyrrg*','optimizing':True,'movingSubset':movingSubset}
    return dotest('hybrid_lazyrrtstar',numTrials,plannerSettings=plannerSettings,maxTime=totalMaxTime,numRestarts=numRestarts,plannerContinueOnRestart=True,plannerMaxIters=200000,plannerMaxTime=totalMaxTime/(2*numRestarts),optimizationMaxIters=optimizationMaxIters*numRestarts)

def doHybridSBL(numTrials=1):
    plannerSettings={'type':'sbl','perturbationRadius':1.0,'bidirectional':True,'movingSubset':movingSubset}
    return dotest('hybrid_sbl',numTrials,plannerSettings=plannerSettings,maxTime=totalMaxTime,numRestarts=numRestarts,plannerMaxIters=200000,plannerMaxTime=totalMaxTime/(numRestarts*2),optimizationMaxIters=numRestarts*optimizationMaxIters)

def doHybridShortcutSBL(numTrials=1):
    plannerSettings={'type':'sbl','perturbationRadius':1.0,'bidirectional':True,'shortcut':True, 'optimizing':True,'movingSubset':movingSubset}
    return dotest('shortcut_hybrid_sbl',numTrials,plannerSettings=plannerSettings,maxTime=totalMaxTime,numRestarts=numRestarts,plannerMaxIters=200000,plannerMaxTime=totalMaxTime/(numRestarts*2),optimizationMaxIters=numRestarts*optimizationMaxIters)

def doAllTests():
    doRRTStar(numAllTrials)
    doLazyRRGStar(numAllTrials)
    doRestartSBL(numAllTrials)
    doRestartShortcutSBL(numAllTrials)
    doHybridRRTStar(numAllTrials)
    doHybridLazyRRGStar(numAllTrials)
    doHybridSBL(numAllTrials)
    doHybridShortcutSBL(numAllTrials)
    print("*** All tests completed, saved to output/robotplanopt_* ***")


lastPlan = None
def runPlanner(runfunc,name):
    global lastPlan
    res = runfunc()
    if res:
        if lastPlan:
            vis.remove(lastPlan)
        vis.add(name,res)
        vis.setColor(name,1,0,0)
        vis.animate(("world",robot.getName()),res)
        lastPlan = name

vis.addAction(lambda: runPlanner(doRRTStar,"RRT*"),"Run RRT*")
vis.addAction(lambda: runPlanner(doHybridRRTStar,"Hybrid RRT*"),"Run Hybrid RRT*")
vis.addAction(lambda: runPlanner(doLazyRRGStar,"Lazy-RRG*"),"Run Lazy-RRG*")
vis.addAction(lambda: runPlanner(doHybridLazyRRGStar,"Hybrid LazyRRG*"),"Run Hybrid LazyRRG*")
vis.addAction(lambda: runPlanner(doRestartSBL,"Restart SBL"),"Run Restart SBL")
vis.addAction(lambda: runPlanner(doHybridSBL,"Hybrid SBL"),"Run Hybrid SBL")
vis.addAction(lambda: runPlanner(doRestartShortcutSBL,"Restart-Shortcut SBL"),"Run Restart-Shortcut SBL")
vis.addAction(lambda: runPlanner(doHybridShortcutSBL,"Hybrid-Shortcut SBL"),"Run Hybrid-Shortcut SBL*")
vis.addAction(doAllTests,"Run all tests")

print("Beginning visualization.")
vis.show()
vis.spin(float('inf'))
vis.kill()