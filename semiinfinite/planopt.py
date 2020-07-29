from __future__ import print_function,division
from klampt.plan import robotplanning
from klampt.model.trajectory import RobotTrajectory
from klampt.math import vectorops
from . import geometryopt
import time

def arc_length_refine(traj,numNewMilestones):
    assert numNewMilestones > 0
    if isinstance(traj,RobotTrajectory):
        metric = traj.robot.distance
    else:
        metric = vectorops.distance
    lengths = [metric(traj.milestones[i],traj.milestones[i+1]) for i in xrange(len(traj.milestones)-1)]
    sumlengths = [0]
    for l in lengths:
        sumlengths.append(sumlengths[-1]+l)
    sumlength = sum(lengths)
    dl = sumlength / numNewMilestones
    alloc = [0]*len(lengths)
    lnext = dl
    for i,l in enumerate(sumlengths):
        while lnext < l:
            alloc[i-1] += 1
            lnext += dl
    newtimes = [traj.times[0]]
    for i,n in enumerate(alloc):
        for j in xrange(n):
            u = float(j+1) / (n+1)
            newtimes.append(traj.times[i] + (traj.times[i+1]-traj.times[i])*u)
        newtimes.append(traj.times[i+1])
    res = traj.remesh(newtimes)[0]
    return res
    

def planOptimizedTrajectory(world,robot,target,maxTime=float('inf'),numRestarts=1,
        plannerSettings={'type':'sbl','shortcut':True},
        plannerMaxIters=3000,
        plannerMaxTime=10.0,
        plannerContinueOnRestart=False,
        optimizationMaxIters=200,
        kinematicsCache=None,
        settings=None,
        logFile=None):
    """
    Args:
        world (WorldModel): contains all objects
        robot (RobotModel): the moving robot
        target: a target configuration
        maxTime (float, optional): a total max time for all planning / optimization calls
        numRestarts (int): number of restarts of the randomized planner
        plannerSettings (dict): a Klamp't planner settings dictionary
        plannerMaxIters (int): max number of iterations to perform each planner call
        plannerMaxTime (float): max time for each planner call
        plannerContinueOnRestart (bool): if true, the planner doesn't get reset every restart
        optimizationMaxIters (int): max number of total optimization iterations
        kinematicsCache (RobotKinematicsCache, optional): modifies the kinematics cache.
        settings (SemiInfiniteOptimizationSettings, optional): modifies the optimizer setting
        logFile (file, optional): if given, this is a CSV file that logs the output

    Returns a RobotTrajectory that (with more iterations) optimizes path length
    """
    bestPath = None
    bestPathCost = float('inf')
    best_instantiated_params = None
    t0 = time.time()
    if kinematicsCache is None:
        kinematicsCache = geometryopt.RobotKinematicsCache(robot)
    if settings is None:
        settings = geometryopt.SemiInfiniteOptimizationSettings()
        settings.minimum_constraint_value = -0.02
    optimizing = False
    if 'optimizing' in plannerSettings:
        optimizing = plannerSettings['optimizing']
        plannerSettings = plannerSettings.copy()
        del plannerSettings['optimizing']
    obstacles = [world.rigidObject(i) for i in xrange(world.numRigidObjects())] + [world.terrain(i) for i in xrange(world.numTerrains())]
    obstaclegeoms = [geometryopt.PenetrationDepthGeometry(obs.geometry(),None,0.04) for obs in obstacles]
    t1 = time.time()
    print("Preprocessing took time",t1-t0)
    if logFile:
        logFile.write("numRestarts,plannerSettings,plannerMaxIters,plannerMaxTime,optimizationMaxIters\n")
        logFile.write("%d,%s,%d,%g,%d\n"%(numRestarts,str(plannerSettings),plannerMaxIters,plannerMaxTime,optimizationMaxIters))
        logFile.write("restart,iterations,time,method,path length\n")

    tstart = time.time()
    startConfig = robot.getConfig()
    for restart in xrange(numRestarts):
        print("***************************************************************")
        #optimize existing best path, if available
        if bestPath is not None and optimizationMaxIters > 0:
            t0 = time.time()
            #optimize the best path
            settings.max_iters = optimizationMaxIters // (numRestarts*2)
            settings.instantiated_params = best_instantiated_params
            print("Optimizing current best path with",settings.max_iters,"iters over",len(bestPath.milestones),"milestones")
            if logFile:
                logFile.write("%d,0,%g,optimize-best,%g\n"%(restart,t0-tstart,bestPathCost))
            trajCache = geometryopt.RobotTrajectoryCache(kinematicsCache,len(bestPath.milestones)-2,qstart=startConfig,qend=target)
            trajCache.tstart = 0
            trajCache.tend = bestPath.times[-1]
            obj = geometryopt.TrajectoryLengthObjective(trajCache)
            constraint = geometryopt.RobotTrajectoryCollisionConstraint(obstaclegeoms,trajCache)
            #constraint = geometryopt.MultiSemiInfiniteConstraint([geometryopt.RobotLinkTrajectoryCollisionConstraint(robot.link(i),obstaclegeoms[0],trajCache) for i in range(robot.numLinks())])
            traj,traj_trace,traj_times,constraintPts = geometryopt.optimizeCollFreeTrajectory(trajCache,bestPath,obstacles,constraints=[constraint],settings=settings)
            if logFile:
                #write the whole trace
                for i,(traji,ti) in enumerate(zip(traj_trace,traj_times)):
                    if i==0: continue
                    logFile.write("%d,%d,%g,optimize-best,%g\n"%(restart,i,t0 + ti - tstart,min(bestPathCost,traji.length())))
            xtraj = trajCache.trajectoryToState(traj)
            #cost = obj.value(xtraj)
            cost = traj.length()
            constraint.setx(xtraj)
            residual = constraint.minvalue(xtraj)
            constraint.clearx()
            print("Optimized cost",cost,"and residual",residual)
            if cost < bestPathCost and residual[0] >= 0:
                print("Got a better cost path by optimizing current best",cost,"<",bestPathCost)
                bestPathCost = cost
                bestPath = traj
                best_instantiated_params = constraintPts
            t1 = time.time()
            if t1 - tstart > maxTime:
                break

        #do sampling-based planningif bestPath is not None:
        if restart == 0 or (not plannerContinueOnRestart and planner.getPath()):
            robot.setConfig(startConfig)
            planner = robotplanning.planToConfig(world,robot,target,**plannerSettings)
        t0 = time.time()
        if logFile:
            logFile.write("%d,0,%g,sample,%g\n"%(restart,t0-tstart,bestPathCost))
        path = None
        oldpath = None
        for it in xrange(0,plannerMaxIters,50):
            oldpath = path
            planner.planMore(50)
            t1 = time.time()
            path = planner.getPath()
            if path:
                if oldpath is None:
                    print("Found a feasible path on restart",restart,"iteration",it)
                path = RobotTrajectory(robot,range(len(path)),path)
                cost = path.length()
                if cost < bestPathCost:
                    print("Got a better cost path from planner",cost,"<",bestPathCost)
                    bestPathCost = cost
                    bestPath = path
                    if logFile:
                        if oldpath is not None:
                            logFile.write("%d,%d,%g,sample-optimize,%g\n"%(restart,it+50,t1-tstart,bestPathCost))
                        else:
                            logFile.write("%d,%d,%g,sample,%g\n"%(restart,it+50,t1-tstart,bestPathCost))
                if not optimizing:
                    break
            if t1 - t0 > plannerMaxTime:
                break
            if t1 - tstart > maxTime:
                break
        if logFile:
            if oldpath is not None:
                logFile.write("%d,%d,%g,sample-optimize,%g\n"%(restart,it+50,t1-tstart,bestPathCost))
            else:
                logFile.write("%d,%d,%g,sample,%g\n"%(restart,it+50,t1-tstart,bestPathCost))
        if t1 - tstart > maxTime:
            break

        #optimize new planned path, if available
        if path and len(path.milestones) == 2:
            #straight line path works, this is certainly optimal
            return path
        elif path and len(path.milestones) > 2:
            #found a feasible path
            t0 = time.time()
            if len(path.milestones) < 30:
                traj0 = arc_length_refine(path,min(30-len(path.milestones),len(path.milestones)))
            else:
                traj0 = path
            traj0.times = [float(i)/(len(traj0.milestones)-1) for i in xrange(len(traj0.times))]
            trajCache = geometryopt.RobotTrajectoryCache(kinematicsCache,len(traj0.milestones)-2,qstart=startConfig,qend=target)
            trajCache.tstart = 0
            trajCache.tend = traj0.times[-1]

            print()
            print("Planned trajectory with",len(traj0.milestones),"milestones")

            obj = geometryopt.TrajectoryLengthObjective(trajCache)
            constraint = geometryopt.RobotTrajectoryCollisionConstraint(obstaclegeoms,trajCache)
            xtraj = trajCache.trajectoryToState(traj0)
            #cost0 = obj.value(xtraj)
            cost0 = traj0.length()
            constraint.setx(xtraj)
            residual0 = constraint.minvalue(xtraj)
            constraint.clearx()
            print("Planned cost",cost0,"and residual",residual0)
            if bestPath:
                settings.max_iters = optimizationMaxIters // (numRestarts*2)
            else:
                settings.max_iters = optimizationMaxIters // (numRestarts)
            settings.instantiated_params = None
            if residual0[0] < 0:
                print("Warning, the planned path has a negative constraint residual?",residual0[0])
                print("  Residual met @ parameter",residual0[1:])
                #raw_input("Warning, the planned path has a negative constraint residual?")
            if optimizationMaxIters > 0:
                if logFile:
                    logFile.write("%d,0,%g,optimize,%g\n"%(restart,t0-tstart,bestPathCost))
                print("Optimizing planned path with",settings.max_iters,"iters")
                traj,traj_trace,traj_times,constraintPts = geometryopt.optimizeCollFreeTrajectory(trajCache,traj0,obstacles,constraints=[constraint],settings=settings)
                if logFile:
                    #write the whole trace
                    for i,(traji,ti) in enumerate(zip(traj_trace,traj_times)):
                        if i==0: continue
                        xtraj = trajCache.trajectoryToState(traji)
                        constraint.setx(xtraj)
                        residual = constraint.minvalue(xtraj)
                        constraint.clearx()
                        if residual[0] >= 0:
                            logFile.write("%d,%d,%g,optimize,%g\n"%(restart,i,t0 + ti - tstart,min(bestPathCost,traji.length())))
                xtraj = trajCache.trajectoryToState(traj)
                #cost = obj.value(xtraj)
                cost = traj.length()
                constraint.setx(xtraj)
                residual = constraint.minvalue(xtraj)
                constraint.clearx()
                print("Optimized cost",cost,"and residual",residual)
                if cost < bestPathCost and residual[0] >= 0:
                    print("Got a better cost path from optimizer",cost,"<",bestPathCost)
                    bestPathCost = cost
                    bestPath = traj
                    best_instantiated_params = constraintPts
                else:
                    print("Optimizer produced a worse cost (",cost,"vs",bestPathCost,") or negative residual, skipping")
                if residual[0] < 0:
                    #raw_input("Warning, the optimized path has a negative constraint residual...")
                    print("Warning, the optimized path has a negative constraint residual...")
                t1 = time.time()
                #if logFile:
                #    logFile.write("%d,%d,%g,optimize,%g\n"%(restart,settings.max_iters,t1-tstart,bestPathCost))
                if t1 - tstart > maxTime:
                    break
        else:
            print("No feasible path was found by the planner on restart",restart)
    return bestPath
