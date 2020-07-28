from __future__ import print_function,division
from klampt import *
from klampt.math import vectorops,so3,se3
from klampt.model.trajectory import Trajectory,RobotTrajectory
from sip import *
import numpy as np
import scipy

try:
    import pyccd
    PYCCD_ENABLED = True
except ImportError:
    PYCCD_ENABLED = False
TEST_PYCCD = False

#whether to populate point clouds with interior points
ADD_INTERIOR_POINTS = False       #adds interior points in the point cloud
USE_BALLS = False                  #associates interior points in the point cloud with balls (can only be True if ADD_INTERIOR_POINTS=True)


def grid_to_numpy(grid):
    vg = grid.getVolumeGrid()
    m,n,p = vg.dims[0],vg.dims[1],vg.dims[2]
    res = np.zeros((m,n,p))
    for i in xrange(m):
        for j in xrange(n):
            for k in xrange(p):
                res[i,j,k] = vg.get(i,j,k)
    return res

def dump_grid_mat(grid,fn='grid.mat'):
    arr = grid_to_numpy(grid)
    slices = dict()
    for i in xrange(arr.shape[0]):
        slices["slice_%d"%(i,)] = arr[i,:,:]
    scipy.io.savemat(fn,slices)

class PenetrationDepthGeometry:
    def __init__(self,geom,gridres=0,pcres=0):
        """Initializes this with a Geometry3D or PenetrationDepthGeometry.  Geometry3D objects
        are converted into dual grid and point cloud form.  The parameters for conversion are
        gridres and pcres.  Either can be set to 0 to disable that representation
        """
        if isinstance(geom,PenetrationDepthGeometry):
            #copy constructor
            self.geom = geom.geom
            self.pc = geom.pc
            self.pcdata = geom.pcdata
            self.grid = geom.grid
            self.polyhedron = geom.polyhedron
        else:
            assert isinstance(geom,Geometry3D)
            self.geom = geom
            self.pc = None
            self.pcdata = None
            self.grid = None
            self.polyhedron = None
            if gridres is not None:
                try:
                    self.grid = geom.convert('VolumeGrid',gridres)
                except Exception as e:
                    print("WARNING: could not convert geometry to VolumeGrid, exception",e)
                    print("May have problems performing normal and distance estimation")
            if pcres is not None:
                self.pc = geom.convert('PointCloud',pcres)
                self.pcdata = self.pc.getPointCloud()
                if ADD_INTERIOR_POINTS:
                    bmin,bmax = self.geom.getBBTight()
                    for i in range(self.pcdata.numPoints()):
                        x = random.uniform(bmin[0],bmax[0])
                        y = random.uniform(bmin[1],bmax[1])
                        z = random.uniform(bmin[2],bmax[2])
                        if self.grid.distance_point([x,y,z]).d < 0:
                            self.pcdata.addPoint([x,y,z])
                    self.pc = Geometry3D(self.pcdata)
            if TEST_PYCCD:
                if not PYCCD_ENABLED:
                    raise ValueError("Can't test PyCCD unless the pyccd module is installed")
                verts = geom.convert('PointCloud',float('inf'))
                vertdata = verts.getPointCloud()
                self.polyhedron = pyccd.Polyhedron()
                self.polyhedron.resize(vertdata.numPoints())
                for i in range(vertdata.numPoints()):
                    self.polyhedron.setVertex(i,*self.pcdata.getPoint(i))
    def setTransform(self,T):
        self.geom.setCurrentTransform(*T)
        if self.pc is not None: self.pc.setCurrentTransform(*T)
        if self.grid is not None: self.grid.setCurrentTransform(*T)
        if self.polyhedron is not None:
            #convert to quaternion in format [x,y,z,w]
            q = so3.quaternion(T[0])
            self.polyhedron.setTransform(T[1],[q[1],q[2],q[3],q[0]])
    def getTransform(self):
        return self.geom.getCurrentTransform()
    def distance(self,other,bound=None):
        """Returns (distance, closestpoint_self, closestpoint_other)"""
        if isinstance(other,(list,tuple)):
            #it's a point (or ball)
            if USE_BALLS:
                center = other[:3]
                radius = other[3]
                other = center
            if bound is not None:
                settings = DistanceQuerySettings()
                settings.upperBound = bound
                res = self.grid.distance_point(other,settings)
            else:
                res = self.grid.distance_point(other)
            assert res.hasClosestPoints,"Need geometry-point distance %s to output closest points"%(self.grid.type(),)
            cp1 = [v for v in res.cp1]
            cp2 = [v for v in res.cp2]
            if USE_BALLS:
                res.d = res.d - radius
                cp1 = cp1 + [0.0]
                cp2 = cp2 + [0.0]
        else:
            settings = DistanceQuerySettings()
            if bound is not None:
                settings.upperBound = bound
            assert isinstance(other,PenetrationDepthGeometry)
            if other.grid is None:
                assert self.grid is not None,"Can't compute fast distance without a signed distance field VolumeGrid / PointCloud"
                assert self.pc is not None,"Can't compute fast distance without a signed distance field VolumeGrid / PointCloud"
                res = self.grid.distance_ext(other.pc,settings)
                print("Uhhh... doing grid to PC instead of the other way around?")
            else:
                res = self.pc.distance_ext(other.grid,settings)
            if bound is not None and res.d >= bound:
                cp1=[0.0]*3
                cp2=[0.0]*3
            else:
                assert res.hasClosestPoints,"Need geometry-geometry distance %s-%s to output closest points"%(self.grid.type(),other.pc.type())
                cp1 = [v for v in res.cp1]
                cp2 = [v for v in res.cp2]
                #check to see if this point is actually on the surface
                if USE_BALLS:
                    #need to return ball, not just surface points
                    #this point might be on the inside, go ahead and shift the closest point
                    dsurf = self.grid.distance_point(cp1)
                    if dsurf.d < 0:
                        res.d += dsurf.d
                        cp1 = cp1 + [-dsurf.d]
                        cp2 = cp2 + [-dsurf.d]
                    else:
                        cp1 = cp1 + [0.0]
                        cp2 = cp2 + [0.0]
        return res.d,cp1,cp2
    def normal(self,pt):
        if USE_BALLS:
            center = pt[:3]
            radius = pt[3]
            pt= center
        res = self.grid.distance_point(pt)
        ptgeom = GeometricPrimitive()
        ptgeom.setPoint(pt)
        if res.hasGradients:
            return [res.grad1[i] for i in range(3)]
        if USE_BALLS:
            raise NotImplementedError("TODO: determine normal with grid-to-ball distance")
        assert res.hasClosestPoints,"Need geometry-point distance to output closest points"
        g = vectorops.sub(res.cp1,res.cp2)
        ptdist = vectorops.norm(g)
        if ptdist <= 1e-8: 
            #TODO: get gradient direction
            #raise NotImplementedError("TODO: get gradient direction at distance=0")
            return [0]*3
        else:
            return vectorops.div(g,ptdist)


class PenetrationGeometryDomain(DomainInterface):
    def __init__(self,g):
        self.g = g
    def sample(self):
        assert not USE_BALLS,"Can't sample penetration geometry domains with balls yet"
        idx = random.randint(0,self.g.pcdata.numPoints()-1)
        return se3.apply(self.g.getTransform(),self.g.pcdata.getPoint(idx))
    def extrema(self):
        assert not USE_BALLS,"Can't get the extrema of a penetration geometry domain with balls yet"
        res = []
        for idx in range(self.g.pcdata.numPoints()):
            res.append(se3.apply(self.g.getTransform(),self.g.pcdata.getPoint(idx)))
        return res



class RobotKinematicsCache:
    def __init__(self,robot,gridres=0.05,pcres=0.02):
        self.robot = robot
        self.geometry = []
        for i in xrange(robot.numLinks()):
            geom = robot.link(i).geometry()
            if geom.empty():
                self.geometry.append(None)
            else:
                self.geometry.append(PenetrationDepthGeometry(geom,gridres,pcres))
        self.dirty = True
    def set(self,q):
        """Updates the robot configuration and PenetrationDepthGeometry transforms"""
        if self.dirty:
            self.robot.setConfig(q)
            for i in xrange(self.robot.numLinks()):
                if self.geometry[i] is None: continue
                self.geometry[i].setTransform(self.robot.link(i).getTransform())
            self.dirty = False
    def clear(self):
        self.dirty = True

class RobotTrajectoryCache:
    """Decodes a RobotTrajectory object from a vector.

    - robot: the RobotModel
    - kinematics: the RobotKinematicsCache
    - numPoints: the number of milestones in the vector. The number of milestones in the Trajectory
      is numPoints, numPoints+1 (one endpoint constrained), or numPoints+1 (two endpoints constrained).
    - times: either False, True, or a list of times.  If False, the RobotTrajectory is
      assumed untimed.  If True, the vector is assumed to carry the numPoints times
      corresponding to the milestones.
    - tstart, tend: the starting / ending times, if times = False or qstart / qend are not None
    - qstart, qend: the optional starting / ending configurations.
    - lipschitzConstants: a matrix of lipschitz constants K[i,j] relating a change in joint space dqi to changes in a link j's geometry's workspace
    - trajectory: the output trajectory
    - linkTransforms: a matrix of the link transforms for each geometry for each milestone
    - interMilestoneLipschitzBounds: the workspace movement bounds between each point on the trajectory
    - dirty: the dirty flag.
    """
    def __init__(self,robot,numPoints,times=False,qstart=None,qend=None):
        """Robot may be either a RobotModel or a RobotKinematicsCache.  In the former case,
        the geometries are initialized with defaults."""
        if isinstance(robot,RobotKinematicsCache):
            self.robot = robot.robot
            self.kinematics = robot
            robot = self.robot
        else:
            self.robot = robot
            self.kinematics = RobotKinematicsCache(robot)
        self.numPoints = numPoints
        self.times = times
        self.qstart = qstart
        self.qend = qend
        self.tstart = 0
        self.tend = 1
        self.trajectory = None
        self.interMilestoneLipschitzBounds = []
        self.lipschitzConstants = np.zeros((self.robot.numLinks(),self.robot.numLinks()))
        radii = np.zeros((self.robot.numLinks(),self.robot.numLinks()))
        ancestors = [[] for i in xrange(self.robot.numLinks())]
        for l in xrange(self.robot.numLinks()):
            link = self.robot.link(l)
            p = link.getParent()
            if p >= 0: 
                ancestors[l] = [p] + ancestors[p]
        qmin,qmax = robot.getJointLimits()
        for l in xrange(self.robot.numLinks()):
            link = self.robot.link(l)
            p = link.getParent()
            if p >= 0: 
                T = link.getParentTransform()
                for a in ancestors[l]:
                    radii[a,l] = radii[a,p] + vectorops.norm(T[1])
                radii[p,l] += vectorops.norm(T[1])
            axis = link.getAxis()
            geom = link.geometry()
            if geom.empty():
                Kgeom = 0
            else:
                #determine a radius of the raw geometry
                Kgeom = 0
                pc = self.kinematics.geometry[l].pcdata
                npoints = pc.numPoints()
                for i in xrange(npoints):
                    k = i*3
                    v = (pc.vertices[k],pc.vertices[k+1],pc.vertices[k+2])
                    if p < 0:
                        #better bound
                        ac = vectorops.cross(v,axis)
                        Kgeom = max(Kgeom,vectorops.norm(ac))
                    else:
                        Kgeom = max(Kgeom,vectorops.norm(v))
            for a in ancestors[l]:
                self.lipschitzConstants[a,l] = radii[a,l] + Kgeom
            self.lipschitzConstants[l,l] = Kgeom
        #print("Robot lipschitz constants",self.lipschitzConstants)
        self.dirty = True
    def set(self,x):
        if self.dirty:
            self.trajectory = self.stateToTrajectory(x)
            m = len(self.trajectory.milestones)
            self.interMilestoneLipschitzBounds = []
            for i in xrange(m-1):
                self.interMilestoneLipschitzBounds.append(self.workspaceMovementBounds(self.trajectory.milestones[i],self.trajectory.milestones[i+1]))
            self.linkTransforms = []
            for i in xrange(m):
                self.robot.setConfig(self.trajectory.milestones[i])
                self.linkTransforms.append([self.robot.link(j).getTransform() for j in xrange(self.robot.numLinks())])
            self.dirty = False
        return
    def clear(self):
        self.dirty = True
    def workspaceMovementBounds(self,q1,q2):
        """Returns a list of upper bounds on the workspace distances of each link's geometry
        when moving between q1 and q2"""
        dq = np.abs(self.robot.interpolateDeriv(q1,q2))
        return np.dot(self.lipschitzConstants.T,dq)
    def workspaceMovementBound(self,link,q1,q2):
        """Returns an upper bound on the workspace distance of the given link's geometry
        when moving between q1 and q2"""
        if isinstance(link,RobotModelLink):
            link = link.index
        dq = np.abs(self.robot.interpolateDeriv(q1,q2)[:link+1])
        return np.dot(self.lipschitzConstants[:link+1,link],dq)
    def stateToTrajectory(self,x):
        """Returns a trajectory that is encoded by the given state"""
        n = self.robot.numLinks()
        m = self.numPoints
        times = []
        milestones = []
        if self.qstart:
            milestones.append(self.qstart)
            if self.times == True or hasattr(self.times,'__iter__'):
                times.append(self.tstart)
        itimeend = 0
        if self.times is True:
            itimeend = m
            times += x[:m]
        assert itimeend+m*n == len(x)
        k = itimeend
        for i in xrange(m):
            assert k+n <= len(x)
            milestones.append(list(x[k:k+n]))
            k += n
        assert k == len(x)
        if hasattr(self.times,'__iter__'):
            times += self.times
        if self.qend:
            milestones.append(self.qend)
            if self.times == True or hasattr(self.times,'__iter__'):
                times.append(self.tend)
        if self.times is False:
            times = [self.tstart + float(i)/(len(milestones)-1)*(self.tend-self.tstart) for i in xrange(len(milestones))]
        return RobotTrajectory(self.robot,times,milestones)
    def trajectoryToState(self,traj):
        """Returns a state x that encodes the given trajectory"""
        if self.qstart is not None:
            assert traj.milestones[0] == self.qstart,"Trajectory needs to match the start configuration"
        if self.qend is not None:
            assert traj.milestones[-1] == self.qend,"Trajectory needs to match the end configuration"
        sshift = (1 if self.qstart is not None else 0)
        eshift = (1 if self.qend is not None else 0)
        assert len(traj.milestones) == self.numPoints + sshift+eshift,"Trajectory needs exactly %d non-fixed milestones"%(self.numPoints,)
        assert traj.times[0] == self.tstart,"Trajectory needs to match the start time"
        assert traj.times[-1] == self.tend,"Trajectory needs to match the end time"
        if self.times is True:
            return traj.times[sshift:-eshift] + sum(traj.milestones[sshift:-eshift],[])
        else:
            return sum(traj.milestones[sshift:-eshift],[])
    def straightLineState(self):
        """Returns a state x that corresponds to a straight line trajectory"""
        assert self.qstart is not None,"Need a start config for a straight line state"
        qend = (self.qend if self.qend is not None else self.qstart)
        sshift = 1
        eshift = (1 if self.qend is not None else 0)
        us = [float(i+sshift)/(self.numPoints+sshift+eshift-1) for i in xrange(self.numPoints)]
        qs = [self.robot.interpolate(self.qstart,qend,u) for u in us]
        ts = ([] if self.times is not True else [self.tstart+u*(self.tend-self.tstart) for u in us])
        return ts + sum(qs,[])



class ObjectPoseObjective(ObjectiveFunctionInterface):
    """An objective function that measures the difference between a pose T and a desired
    pose Tdes. 

    f(T) = weight*||T.t - Tdes.t||^2 + rotationWeight*eR(T.R,Tdes.R)^2

    where eR is the absolute rotation distance.
    """
    def __init__(self,Tdes,weight=1.0,rotationWeight=1.0):
        self.Tdes = Tdes
        self.weight = weight
        self.rotationWeight = rotationWeight
    def value(self,T):
        err = se3.error(self.Tdes,T)
        return self.weight*(vectorops.normSquared(err[3:]) + self.rotationWeight**2*vectorops.normSquared(err[0:3]))
    def minstep(self,T):
        return se3.error(self.Tdes,T)
    def hessian(self,T):
        w = self.weight
        return np.array([w*self.rotationWeight**2]*3+[w]*3)
    def integrate(self,T,dT):
        return (so3.mul(so3.from_rotation_vector(dT[:3]),T[0]),vectorops.add(T[1],dT[3:]))


class ObjectCollisionConstraint(SemiInfiniteConstraintInterface):
    """Returns signed-distance(T,pt) with pt in R^3 in the environment's geometry and T in SE(3).

    Note: requires geometries to support distance queries

    Note: modifies the object's geometry transform on each call.
    """
    def __init__(self,obj,env):
        self.obj = obj
        self.objgeom = PenetrationDepthGeometry(obj if isinstance(obj,(Geometry3D,PenetrationDepthGeometry)) else obj.geometry())
        self.env = PenetrationDepthGeometry(env)
    def setx(self,T):
        self.objgeom.setTransform(T)
    def value(self,T,pt):
        return self.objgeom.distance(pt)[0]
    def minvalue(self,T,bound=None):
        dmin,cp1,cp2 = self.env.distance(self.objgeom,bound)
        if bound is not None and dmin >= bound: return []
        return dmin,cp1
    def df_dx(self,T,pt):
        worlddir = self.objgeom.normal(pt)
        cporiented = so3.apply(T[0],vectorops.sub(pt[:3],T[1]))
        av = worlddir
        aw = vectorops.cross(cporiented,worlddir)
        return np.array(list(aw)+list(av))
    def domain(self):
        return PenetrationGeometryDomain(self.env)

class ObjectConvexCollisionConstraint(ConstraintInterface):
    """Returns signed-distance(T*A,B) where x=T"""
    def __init__(self,objA,objB):
        if not TEST_PYCCD:
            raise ValueError("Can't do convex-convex collisions unless TEST_PYCCD is enabled")
        self.objA = PenetrationDepthGeometry(objA if isinstance(objA,(Geometry3D,PenetrationDepthGeometry)) else objA.geometry())
        self.objB = PenetrationDepthGeometry(objB)
        self.res = None
    def setx(self,T):
        self.objA.setTransform(T)
        self.res = pyccd.signedDistance(self.objA.polyhedron,self.objB.polyhedron)
    def value(self,T):
        sd,sdir,spt = self.res
        return sd
    def df_dx(self,T):
        sd,sdir,spt = self.res
        worlddir = vectorops.mul(sdir,-1.0)
        cporiented = so3.apply(T[0],vectorops.sub(spt,T[1]))
        av = worlddir
        aw = vectorops.cross(cporiented,worlddir)
        return np.array(list(aw)+list(av))

class RobotConfigObjective(ObjectiveFunctionInterface):
    """An objective function that returns the squared distance between a configuration
    and qdes, i.e., f(q) = weight*||q-qdes||^2.
    """
    def __init__(self,robot,qdes,weight=1.0):
        self.robot = robot
        self.qdes = qdes
        self.weight = weight
    def value(self,q):
        return self.weight*self.robot.distance(q,self.qdes)**2
    def minstep(self,q):
        return self.robot.interpolateDeriv(q,self.qdes)

class RobotLinkCollisionConstraint(SemiInfiniteConstraintInterface):
    """Represents collisions between a single robot link and a single static geometry

    Note: requires geometries to support distance queries

    Note: modifies the robot's config on each call.
    """
    def __init__(self,link,env,robotcache=None):
        self.link = link
        self.robot = robotcache if robotcache is not None else RobotKinematicsCache(link.robot())
        self.env = PenetrationDepthGeometry(env)
    def setx(self,q):
        self.robot.set(q)
    def clearx(self):
        self.robot.clear()
    def value(self,q,pt):
        return self.robot.geometry[self.link.index].distance(pt)[0]
    def minvalue(self,q,bound=None):
        dmin,cp1,cp2 = self.env.distance(self.robot.geometry[self.link.index],bound)
        if bound is not None and dmin >= bound: return []
        return dmin,cp1
    def df_dx(self,q,pt):
        localpt = self.link.getLocalPosition(pt[:3])
        worlddir = self.robot.geometry[self.link.index].normal(pt)
        Jp = self.link.getPositionJacobian(localpt)
        return np.dot(np.array(Jp).T,worlddir)
    def domain(self):
        return PenetrationGeometryDomain(self.env)

class RobotLinkConvexCollisionConstraint(ConstraintInterface):
    """Represents collisions between a single robot link and a single static geometry

    Note: requires PyCCD

    Note: modifies the robot's config on each call.
    """
    def __init__(self,link,env,robotcache=None):
        self.link = link
        self.robot = robotcache if robotcache is not None else RobotKinematicsCache(link.robot())
        self.env = PenetrationDepthGeometry(env)
    def setx(self,q):
        self.robot.set(q)
        self.res = pyccd.signedDistance(self.robot.geometry[self.link.index].polyhedron,self.env.polyhedron)
    def clearx(self):
        self.robot.clear()
    def value(self,q):
        sd,sdir,spt = self.res
        return sd
    def df_dx(self,q):
        sd,sdir,spt = self.res
        localpt = self.link.getLocalPosition(spt)
        worlddir = vectorops.mul(sdir,-1.0)
        Jp = self.link.getPositionJacobian(localpt)
        return np.dot(np.array(Jp).T,worlddir)

class TrajectoryLengthObjective(ObjectiveFunctionInterface):
    def __init__(self,trajcache):
        self.trajcache = trajcache
        self.hessian_cache = None
        self.xdes = np.asarray(self.trajcache.straightLineState())
    def value(self,x):
        self.trajcache.set(x)
        l = 0
        for i in xrange(len(self.trajcache.trajectory.milestones)-1):
            l += self.trajcache.robot.distance(self.trajcache.trajectory.milestones[i],self.trajcache.trajectory.milestones[i+1])**2
        self.trajcache.clear()
        return l
    def hessian(self,x):
        if self.hessian_cache is None:
            n = self.trajcache.robot.numLinks()
            sshift = (1 if self.trajcache.qstart is not None else 0)
            eshift = (1 if self.trajcache.qend is not None else 0)
            nfree = len(self.trajcache.trajectory.milestones)-sshift-eshift
            assert nfree == self.trajcache.numPoints
            blocks = [[None]*nfree for i in xrange(nfree)]
            for i in xrange(sshift,sshift+self.trajcache.numPoints):
                k = i-sshift
                if i == 0:
                    #first block, free endpoint
                    blocks[k][k] = 2*scipy.sparse.eye(n)
                    blocks[k][k+1] = -2*scipy.sparse.eye(n)
                elif i+1 == len(self.trajcache.trajectory.milestones):
                    #last block, free endpoint
                    blocks[k][k] = 2*scipy.sparse.eye(n)
                    blocks[k][k-1] = -2*scipy.sparse.eye(n)
                else:
                    blocks[k][k] = 4*scipy.sparse.eye(n)
                    if k > 0:
                        blocks[k][k-1] = -2*scipy.sparse.eye(n)
                    if k+1 < nfree:
                        blocks[k][k+1] = -2*scipy.sparse.eye(n)
            self.hessian_cache = scipy.sparse.bmat(blocks,format='csc')
        return self.hessian_cache
    def minstep(self,x):
        #optimum is linear spacing
        return (self.xdes-x)*0.75

def intervalLipschitzMinimum(a,b,K):
    """Given a function f on interval [0,1] satisfying f(0)=a, f(1)=b, and |f'(x)|<=K for all
    x in [0,1], returns a lower bound on f over [0,1]"""
    #simple bound min(a - 0.5*K,b-0.5*K)
    #better bound solve min_u max(a - u*K,b-(1-u)*K)
    #(a - b + K)/2K = u
    if K == 0: return min(a,b)
    u = (a-b+K)/(2*K)
    if u < 0 or u > 1: return min(a-0.5*K,b-0.5*K)
    return a-u*K

class RobotLinkTrajectoryCollisionConstraint(SemiInfiniteConstraintInterface):
    """Represents collisions along a trajectory between a single robot link and a
    single static geometry.

    The 4-parameter is [t]+pt where pt is the point in world coordinates

    Note: modifies the robot's config on each call.
    """
    def __init__(self,link,env,trajectorycache):
        self.link = link
        self.env = env
        self.trajectory = trajectorycache
        self.verbose = 0
        assert isinstance(trajectorycache,RobotTrajectoryCache)
    def setx(self,x):
        self.trajectory.set(x)
    def clearx(self):
        self.trajectory.clear()
    def value(self,x,t_pt):
        t = t_pt[0]
        worldpt = t_pt[1:4]
        if USE_BALLS:
            #include the radius
            worldpt = t_pt[1:5]
        self.trajectory.kinematics.set(self.trajectory.trajectory.eval(t))
        geom = self.trajectory.kinematics.geometry[self.link.index]
        res = geom.distance(worldpt)
        self.trajectory.kinematics.clear()
        return res[0]
    def minvalue(self,x,bound=None):
        t0 = time.time()
        traj = self.trajectory.trajectory
        robot = self.trajectory.robot
        geom = self.trajectory.kinematics.geometry[self.link.index]
        dmin = float('inf') if bound is None else bound
        tmin = None
        dmilestones = []
        for i in xrange(len(traj.milestones)):
            #self.trajectory.robot.setConfig(q)
            geom.setTransform(self.trajectory.linkTransforms[i][self.link.index])
            dbnd = 1000
            if i > 0:
                dbnd = self.trajectory.interMilestoneLipschitzBounds[i-1][self.link.index]
            if i+1 < len(traj.milestones):
                dbnd = min(dbnd,self.trajectory.interMilestoneLipschitzBounds[i][self.link.index])
            di = self.env.distance(geom,dmin+dbnd*0.5)[0]
            if self.verbose >= 2: print("Distance at milestone",i,"is",di)
            if di < dmin:
                dmin = di
                tmin = traj.times[i]
            dmilestones.append(di)
        if self.verbose >= 1: print("At milestones, minimum distance is",dmin,"at time",tmin)
        epsilon = 1e-2
        edgequeue = []
        for i in xrange(len(traj.milestones)-1):
            di = dmilestones[i]
            dn = dmilestones[i+1]
            dbnd = self.trajectory.interMilestoneLipschitzBounds[i][self.link.index]
            if self.verbose >= 2: print("Lipschitz bound on edge",i,"is",dbnd)
            dlow = intervalLipschitzMinimum(di,dn,dbnd)
            if dlow < dmin:
                #optimum could be in this edge range, subdivide
                heapq.heappush(edgequeue,(dlow,(i,0,1,di,dn)))
        if self.verbose >= 1: print("Original edge queue distances",[(info[0],info[1][0]) for info in edgequeue])
        numsubdivisions = 0
        while len(edgequeue) > 0:
            dlow,(edge,u,v,du,dv) = heapq.heappop(edgequeue)
            if dlow >= dmin: continue  #no possible minimum on this segment
            numsubdivisions += 1
            midpt = (u+v)*0.5
            q = traj.interpolate(traj.milestones[edge],traj.milestones[edge+1],midpt,traj.times[edge+1]-traj.times[edge])
            robot.setConfig(q)
            geom.setTransform(self.link.getTransform())
            dbnd = self.trajectory.interMilestoneLipschitzBounds[edge][self.link.index]
            duration = (v - u)*0.25
            di = self.env.distance(geom,dmin+dbnd*duration*2)[0]
            if di < dmin:
                dmin = di
                tmin = traj.times[edge] + midpt*(traj.times[edge+1]-traj.times[edge])
            #try recrusing with halves
            if duration > epsilon:
                dlow = intervalLipschitzMinimum(du,di,duration*dbnd)
                if dlow < dmin:
                    heapq.heappush(edgequeue,(dlow,(edge,u,midpt,du,di)))
                dlow = intervalLipschitzMinimum(di,dv,duration*dbnd)
                if dlow < dmin:
                    heapq.heappush(edgequeue,(dlow,(edge,midpt,v,di,dv)))
        t1 = time.time()
        if self.verbose >= 1: print("Performed",numsubdivisions,"subdivision steps","in time",t1-t0)
        if tmin == None:
            #bound killed everything
            return []
        robot = self.trajectory.robot
        robot.setConfig(self.trajectory.trajectory.eval(tmin))
        geom.setTransform(self.link.getTransform())
        d,cp1,cp2 = self.env.distance(geom)
        return dmin,(tmin,)+tuple(cp1)
    def df_dx(self,x,t_pt):
        t = t_pt[0]
        pt = t_pt[1:4]
        if USE_BALLS:
            #include the radius
            pt = t_pt[1:5]
        geom = self.trajectory.kinematics.geometry[self.link.index]
        worlddir = geom.normal(pt)
        localpt = self.link.getLocalPosition(pt[:3])
        Jp = self.link.getPositionJacobian(localpt)
        return np.dot(np.array(Jp).T,worlddir)
    def domain(self):
        return CartesianProductDomain(IntervalDomain(self.trajectory.tstart,self.trajectory.tend),
                                        PenetrationGeometryDomain(self.env))


class RobotTrajectoryCollisionConstraint(SemiInfiniteConstraintInterface):
    """Represents collisions along a trajectory between an entire robot and one or more
    static geometries.

    The 6-parameter is [linkindex,envindex,t]+pt where pt is the point in world coordinates

    Note: modifies the robot's config on each call.
    """
    def __init__(self,envs,trajectorycache):
        assert isinstance(trajectorycache,RobotTrajectoryCache)
        self.robot = trajectorycache.robot
        self.envs = envs
        self.trajectory = trajectorycache
        self.verbose = 0
        self.epsilon = 1e-2
        self.activeLinks = []
        qmin,qmax = self.trajectory.robot.getJointLimits()
        for i in xrange(self.trajectory.robot.numLinks()):
            if qmin[i] < qmax[i]:
                self.activeLinks.append(i)
            else:
                print("RobotTrajectoryCollisionConstraint: ignoring link",i)
    def setx(self,x):
        self.trajectory.set(x)
    def clearx(self):
        self.trajectory.clear()
    def value(self,x,y):
        link = y[0]
        env = y[1]
        t = y[2]
        worldpt = y[3:6]
        if USE_BALLS:
            #include the radius
            worldpt = y[3:7]
        q = self.trajectory.trajectory.eval(t)
        self.trajectory.kinematics.set(q)
        geom = self.trajectory.kinematics.geometry[link]
        res = geom.distance(worldpt)
        """
        if link == 4:
            print("Value at time",t,"link",link,"env",env,"worldpt",worldpt,"is",res[0])
            print("  config norm",vectorops.norm(q),"traj norm",vectorops.norm(x))
        """
        self.trajectory.kinematics.clear()
        return res[0]
    def minvalue(self,x,bound=None):
        distance_bound_inflation = 1e-3
        t0 = time.time()
        traj = self.trajectory.trajectory
        robot = self.trajectory.robot
        dmin = float('inf') if bound is None else bound
        linkcheckhist = [0]*robot.numLinks()
        envcheckhist = [0]*len(self.envs)
        linkmin = None
        envmin = None
        tmin = None
        qmin = None
        active_milestones = [dict() for i in xrange(len(traj.milestones))]
        for link in self.activeLinks[::-1]:
            geom = self.trajectory.kinematics.geometry[link]
            for i in xrange(len(traj.milestones)):
                #self.trajectory.robot.setConfig(q)
                geom.setTransform(self.trajectory.linkTransforms[i][link])
                dbnd = 1000
                if i > 0:
                    dbnd = self.trajectory.interMilestoneLipschitzBounds[i-1][link]
                if i+1 < len(traj.milestones):
                    dbnd = min(dbnd,self.trajectory.interMilestoneLipschitzBounds[i][link])
                dimin = dmin + dbnd*0.5 + distance_bound_inflation
                for j,env in enumerate(self.envs):
                    dij = env.distance(geom,dimin)[0]
                    linkcheckhist[link] += 1
                    envcheckhist[j] += 1
                    if dij < dimin:
                        dimin = dij
                        active_milestones[i][link,j] = dij
                    if dij < dmin:
                        dmin = dij
                        tmin = traj.times[i]
                        qmin = traj.milestones[i]
                        envmin = j
                        linkmin = link
        if self.verbose>=1: print("Among all milestones, minimum distance is",dmin,"on link",linkmin,"at time",tmin)
        edgequeue = []
        for i in xrange(len(traj.milestones)-1):
            di = active_milestones[i]
            dn = active_milestones[i+1]
            dbnd = self.trajectory.interMilestoneLipschitzBounds[i]
            #print("Lipschitz bound on edge",i,"is",dbnd)
            active = dict()
            for (link,env),d in di.iteritems():
                if (link,env) in dn:
                    dlow = intervalLipschitzMinimum(d,dn[link,env],dbnd[link])
                    if dlow < dmin:
                        active[link,env] = dlow
                else:
                    dlow = d - dbnd[link]*0.5
                    if dlow < dmin:
                        active[link,env] = dlow
            for (link,env),d in dn.iteritems():
                if (link,env) not in di:
                    dlow = d - dbnd[link]*0.5
                    if dlow < dmin:
                        active[link,env] = dlow
            if len(active) > 0:
                if self.verbose>=2: 
                    alinks = [float('inf')]*robot.numLinks()
                    aenvs = [float('inf')]*len(self.envs)
                    for (link,env),d in active.iteritems():
                        if d < alinks[link]:
                            alinks[link] = d
                        if d < aenvs[env]:
                            aenvs[env] = d
                    for link,d in enumerate(alinks):
                        if d < 1000:
                            print("At edge",i,"link",link,"is active with distance lower bound",d)
                    for env,d in enumerate(aenvs):
                        if d < 1000:
                            print("At edge",i,"env",env,"is active with distance lower bound",d)
                heapq.heappush(edgequeue,(min(active.values()),active,(i,0,1,di,dn)))
        #print("Original edge queue distances",[(info[0],info[2][0]) for info in edgequeue]
        if self.verbose>=1: print("Further subdividing",len(edgequeue),"edges")
        numsubdivisions = 0
        numsubdivisionchecks = 0
        while len(edgequeue) > 0:
            dlowmin,dlow,(edge,u,v,du,dv) = heapq.heappop(edgequeue)
            if dlowmin >= dmin: continue  #no possible minimum on this segment
            numsubdivisions += 1
            midpt = (u+v)*0.5
            assert 0 < midpt < 1
            duration = (v - u)
            q = traj.interpolate(traj.milestones[edge],traj.milestones[edge+1],midpt,traj.times[edge+1]-traj.times[edge])
            t = traj.times[edge] + midpt*(traj.times[edge+1]-traj.times[edge])
            #TEST
            #qtest = traj.eval(t)
            #assert vectorops.distance(q,qtest)<1e-7,"Uh... evaluation of trajectory at time %g does not equal interpolated config? Deviation %g"%(t,vectorops.distance(q,qtest))
            self.trajectory.kinematics.set(q)
            #list of (d,linkindex,envindex) triples
            active = dict()
            for (link,env),d in dlow.iteritems():
                if d >= dmin: continue
                geom = self.trajectory.kinematics.geometry[link]
                dbnd = self.trajectory.interMilestoneLipschitzBounds[edge][link]
                numsubdivisionchecks += 1
                dij = self.envs[env].distance(geom,dmin+dbnd*duration*0.5 + distance_bound_inflation)[0]
                if self.verbose >= 2:
                    print("Dist %d - %d time %g: %g (bound %g)"%(link,env,t,dij,dmin+dbnd*duration*0.5 + distance_bound_inflation))
                linkcheckhist[link] += 1
                envcheckhist[env] += 1
                if dij < dmin+dbnd*duration*0.5:
                    active[link,env] = dij
                if dij < dmin:
                    dmin = dij
                    tmin = traj.times[edge] + midpt*(traj.times[edge+1]-traj.times[edge])
                    qmin = q
                    linkmin = link
                    envmin = env
            self.trajectory.kinematics.clear()
            #try recursing with halves
            duration *= 0.5
            if len(active) > 0 and duration > self.epsilon:
                dbnd = self.trajectory.interMilestoneLipschitzBounds[edge]
                c1 = dict()
                c2 = dict()
                for (link,env),d in active.iteritems():
                    if (link,env) in du:
                        dlow = intervalLipschitzMinimum(du[link,env],d,duration*dbnd[link])
                    else:
                        dlow = d-dbnd[link]*duration*0.5
                    if dlow < dmin:
                        c1[link,env] = dlow
                    if (link,env) in dv:
                        dlow = intervalLipschitzMinimum(d,dv[link,env],duration*dbnd[link])
                    else:
                        dlow = d-dbnd[link]*duration*0.5
                    if dlow < dmin:
                        c2[link,env] = dlow
                if len(c1) > 0:
                    heapq.heappush(edgequeue,(min(c1.values()),c1,(edge,u,midpt,du,active)))
                if len(c2) > 0:
                    heapq.heappush(edgequeue,(min(c2.values()),c2,(edge,midpt,v,active,dv)))
        t1 = time.time()
        if self.verbose>=1: 
            print("Performed",numsubdivisions,"subdivision steps including",numsubdivisionchecks,"collision checks in time",t1-t0)
            print("# of link checks:",linkcheckhist)
            print("# of env checks:",envcheckhist)
            print("Closest point distance",dmin,"on link",linkmin,"object",envmin,"at time",tmin)
            """
            #TEST brute force
            dmin_bf = float('inf')
            tmin_bf = None
            linkmin_bf = None
            envmin_bf = None
            for i in xrange(1001):
                u = float(i)/1000.0
                t = self.trajectory.trajectory.times[0] + u*self.trajectory.trajectory.duration()
                q = self.trajectory.trajectory.eval(t)
                self.trajectory.kinematics.set(q)
                for link in range(robot.numLinks()):
                    geom = self.trajectory.kinematics.geometry[link]
                    for j,env in enumerate(self.envs):
                        dij = env.distance(geom)[0]
                        if dij < dmin_bf:
                            dmin_bf = dij
                            tmin_bf = t 
                            linkmin_bf = link
                            envmin_bf = j
                self.trajectory.kinematics.clear()
            print("Brute force minimum distance",dmin_bf,"on link",linkmin_bf,"object",envmin_bf,"at time",tmin_bf)
            raw_input()
            """
        if tmin == None:
            #bound killed everything
            return []
        assert tmin != None
        robot.setConfig(qmin)
        geom = self.trajectory.kinematics.geometry[linkmin]
        geom.setTransform(robot.link(linkmin).getTransform())
        d,cp1,cp2 = self.envs[envmin].distance(geom)
        """
        print(".minvalue Config",qmin,"and environment point",cp1,"with local point",cp2)
        print("  Geomtransform",geom.geom.getCurrentTransform())
        d_test,cp1_test,cp2_test = geom.distance(cp1)
        print("testing %g = %g = %g has abs value %g"%(dmin,d,d_test,vectorops.distance(cp1,cp2)))
        print("testing",cp1,"=",cp1_test)
        print("testing",cp2,"=",cp2_test)
        """
        return dmin,(linkmin,envmin,tmin)+tuple(cp1)
    def df_dx(self,x,y):
        #return self.df_dx_numeric(x,y,1e-3)
        link = y[0]
        env = y[1]
        t = y[2]
        pt = y[3:6]
        if USE_BALLS:
            #include the radius
            pt = y[3:7]
        robot = self.robot
        q = self.trajectory.trajectory.eval(t)
        self.trajectory.kinematics.set(q)

        geom = self.trajectory.kinematics.geometry[link]
        worlddir = geom.normal(pt)
        self.trajectory.kinematics.clear()

        localpt = robot.link(link).getLocalPosition(pt[:3])
        Jp = robot.link(link).getPositionJacobian(localpt)
        dq = np.dot(np.array(Jp).T,worlddir)
        i,u = self.trajectory.trajectory.getSegment(t)
        m = self.trajectory.numPoints
        n = robot.numLinks()
        sshift = (1 if self.trajectory.qstart is not None else 0)
        eshift = (1 if self.trajectory.qend is not None else 0)
        i -= sshift

        qs = []
        inds = []
        #dx = np.zeros(m*n)
        if i >= 0 and i+1 <= m:
            #dx[i*n:i*n+n] = (1-u)*dq
            qs.append((1-u)*dq)
            inds += list(range(i*n,i*n+n))
        if i >= 0 and i+1 < m:
            #dx[i*n+n:i*n+2*n] = u*dq
            qs.append(u*dq)
            inds += list(range(i*n+n,i*n+2*n))
        if len(qs) == 0:
            return scipy.sparse.csr_matrix((qs,[],[0,0]),shape=(1,m*n))
        qs = np.hstack(qs)
        for i in inds:
            assert i >= 0 and i < m*n,"Failed to calculate proper df/dx for link %d, env %d, t %f"%(link,env,t)
        dx = scipy.sparse.csr_matrix((qs,inds,[0,len(inds)]),shape=(1,m*n))
        return dx
        
    def domain(self):
        return CartesianProductDomain(SetDomain(range(self.robot.numLinks())),
                                        SetDomain(range(len(self.envs))),
                                        IntervalDomain(self.trajectory.tstart,self.trajectory.tend),
                                        UnionDomain([PenetrationGeometryDomain(e) for e in self.envs]))


def makeCollisionConstraints(obj,envs,gridres=0,pcres=0):
    """Makes the most efficient collision checker between the given object (RobotModel,
    RobotModelLink, list of RobotModelLinks, RigidObjectModel, or Geometry3D)
    and one or more static geometries.

    Return value is (constraints,pairs) where constraints is the list of constraints, and pairs
    is a corresponding list of paired entities from the obj, envs arguments.

    If you have a large number of robot links / objects / envs, the optimizer may be
    faster if you put them into a single MultiSemiInfiniteConstraint.  However, during
    debugging you will need to do more work when parsing the constraint value since it's
    a single number rather than a vector of elements corresponding to collision pairs.
    """
    if not hasattr(envs,'__iter__'):
        envs = [envs]
    print("Calling makeCollisionConstraints",obj.__class__.__name__)
    envgeoms = [None]*len(envs)
    for i,e in enumerate(envs):
        if isinstance(e,PenetrationDepthGeometry):
            envgeoms[i] = e
        else:
            if isinstance(e,Geometry3D):
                envgeoms[i] = e
                print("  Converting environment plain geometry",i)
            else:
                envgeoms[i] = e.geometry()
                print("  Converting environment item",e.getName())
            envgeoms[i] = PenetrationDepthGeometry(envgeoms[i],gridres,pcres)
    if isinstance(obj,RobotModel):
        #everything shares the same robot cache
        cache = RobotKinematicsCache(obj,gridres,pcres)
        return makeCollisionConstraints(cache,envs,gridres,pcres)
    elif isinstance(obj,RobotKinematicsCache):
        #everything shares the same robot cache
        robot = obj.robot
        res = []
        pairs = []
        for i in xrange(robot.numLinks()):
            link = robot.link(i)
            if link.geometry().empty(): continue
            if TEST_PYCCD:
                res += [RobotLinkConvexCollisionConstraint(link,e,obj) for e in envgeoms]
            else:
                res += [RobotLinkCollisionConstraint(link,e,obj) for e in envgeoms]
            pairs += [(link,e) for e in envs]
        return res,pairs
    elif isinstance(obj,RobotModelLink):
        cache = RobotKinematicsCache(obj.robot(),gridres,pcres)
        if TEST_PYCCD:
            return [RobotLinkConvexCollisionConstraint(obj,e,cache) for e in envgeoms],[(obj,e) for e in envs]
        else:
            return [RobotLinkCollisionConstraint(obj,e,cache) for e in envgeoms],[(obj,e) for e in envs]
    elif hasattr(obj,'__iter__'):
        assert all(isinstance(o,RigidObjectModel) for o in obj),"If a list of objects is given, it must be a list of RobotModelLinks"
        if len(obj) == 0: return []
        cache = RobotKinematicsCache(obj[0].robot(),gridres,pcres)
        res = []
        pairs = []
        for o in obj:
            if TEST_PYCCD:
                res += [RobotLinkConvexCollisionConstraint(o,e,cache) for e in envgeoms]
            else:
                res += [RobotLinkCollisionConstraint(o,e,cache) for e in envgeoms]
            pairs += [(o,e) for e in envs]
        return res,pairs
    elif isinstance(obj,RigidObjectModel):
        geom = PenetrationDepthGeometry(obj.geometry(),gridres,pcres)
        if TEST_PYCCD:
            return [ObjectConvexCollisionConstraint(geom,e) for e in envgeoms],[(obj,e) for e in envs]
        else:
            return [ObjectCollisionConstraint(geom,e) for e in envgeoms],[(obj,e) for e in envs]
    else:
        assert isinstance(obj,Geometry3D),"obj must be a robot, link, rigid object, or geometry"
        geom = PenetrationDepthGeometry(obj,gridres,pcres)
        if TEST_PYCCD:
            return [ObjectConvexCollisionConstraint(geom,e) for e in envgeoms],[(obj,e) for e in envs]
        else:
            return [ObjectCollisionConstraint(geom,e) for e in envgeoms],[(obj,e) for e in envs]        


def optimizeCollFree(obj,env,Tinit,Tdes=None,
    settings=None,verbose=1,
    want_trace=True,want_times=True,want_constraints=True):
    """Uses the optimizeSemiInfinite function to optimize the transform of object obj in environment env
    so it is collision-free.

    Parameters:
    - obj: the Klamp't RigidObjectModel or Geometry3D for the object
    - env: the Klamp't RigidObjectModel, TerrainModel, or Geometry3D for the static object
    - Tinit: the initial Klamp't se3 transform of the object
    - Tdes: the desired Klamp't se3 transform of the object.  If None, uses Tinit as the goal
    - settings: a SemiInfiniteOptimizationSettings object to customize solver settings
    - verbose: controls how much output you want to see
    - want_trace,want_times,want_constraints: set to True if you want to return the
      trace, times, and/or constraints (see below)

    Returns:
        If want_trace, want_times, and want_constraints are all false, returns the resulting pose T.
        Otherwise, returns a tuple (T,...) optionally containing:
        - trace: the list of transforms at each major iteration
        - times: the list of major iteration times
        - constraints: a list containing the final set of active constraint points
    """
    if Tdes is None:
        Tdes = Tinit
    objective = ObjectPoseObjective(Tdes)
    if TEST_PYCCD:
        constraint = ObjectConvexCollisionConstraint(obj,env)
    else:
        constraint = ObjectCollisionConstraint(obj,env)
    res = optimizeSemiInfinite(objective,[constraint],Tinit,verbose=verbose,settings=settings)

    #format the output
    if not want_trace and not want_times and not want_constraint_pts:
        return res.x
    retlist = [res.x]
    if want_trace:
        retlist.append(res.trace)
    if want_times:
        retlist.append(res.trace_times)
    if want_constraints:
        retlist.append(res.instantiated_params[0])
    return retlist

def optimizeCollFreeMinDist(obj,env,Tinit,Tdes=None,
    settings=None,verbose=1,
    want_trace=True,want_times=True,want_constraints=True):
    """Uses the generic optimize function and a minimum constraint adaptor

    Parameters:
    - obj: the Klamp't RigidObjectModel or Geometry3D for the object
    - env: the Klamp't RigidObjectModel, TerrainModel, or Geometry3D for the static object
    - Tinit: the initial Klamp't se3 transform of the object
    - Tdes: the desired Klamp't se3 transform of the object.  If None, uses Tinit as the goal
    - settings: a SemiInfiniteOptimizationSettings object to customize solver settings
    - verbose: controls how much output you want to see
    - want_trace,want_times,want_constraints: set to True if you want to return the
      trace, times, and/or constraints (see below)

    Returns:
        If want_trace, want_times, and want_constraints are all false, returns the resulting pose T.
        Otherwise, returns a tuple (T,...) optionally containing:
        - trace: the list of transforms at each major iteration
        - times: the list of major iteration times
        - constraints: an empty list, only here to be compatible with optimizeCollFree
    """
    if Tdes is None:
        Tdes = Tinit
    objective = ObjectPoseObjective(Tdes)
    semi_inf_constraint = ObjectCollisionConstraint(obj,env)
    constraint = MinimumConstraintAdaptor(semi_inf_constraint)
    res = optimizeStandard(objective,[constraint],Tinit,verbose=verbose,settings=settings)

    #format the output
    if not want_trace and not want_times and not want_constraint_pts:
        return res.x
    retlist = [res.x]
    if want_trace:
        retlist.append(res.trace)
    if want_times:
        retlist.append(res.trace_times)
    if want_constraints:
        retlist.append([])
    return retlist


def optimizeCollFreeRobot(robot,env,qdes=None,qinit=None,objective=None,constraints=None,
    settings=None,verbose=1,
    want_trace=True,want_times=True,want_constraints=True):
    """

    Parameters:
    - robot: the Klamp't RobotModel
    - env: a Klamp't RigidObjectModel, TerrainModel, or Geometry3D for the static object.  Can also be
      a list of static objects.
    - qdes: the desired configuration of the robot.  Can be None, in which case the initial configuration
      is used.  None is not compatible with qinit = 'random' or 'random-collision-free'
    - qinit: the initial configuration of the robot.  Can be a list, None, 'random', or 'random-collision-free'
    - objective: if None, the objective is to arrive at qdes.  Otherwise, an ObjectiveFunctionInterface
      that you would like to minimize.
    - constraints: if None, the constraints are created using makeCollisionConstraints(robot,env).  If you want
      to save some overhead over multiple calls, create the constraints yourself and pass them in here.
    - settings: a SemiInfiniteOptimizationSettings object to customize solver settings
    - verbose: controls how much output you want to see
    - want_trace,want_times,want_constraints: set to True if you want to return the
      trace, times, and/or constraints (see below)

    Returns:
        If want_trace, want_times, and want_constraints are all false, returns the resulting configuration q.
        Otherwise, returns a tuple (q,...) optionally containing:
        - trace: the list of configurations at each major iteration
        - times: the list of major iteration times
        - constraints: a list containing the final set of active constraint points
    """
    if settings is None:
        settings = SemiInfiniteOptimizationSettings()
        settings.minimum_constraint_value = -0.02
    if constraints is None:
        constraints,pairs = makeCollisionConstraints(robot,env)
    qmin,qmax = robot.getJointLimits()

    #determine initial and desired configurations
    if qinit is None:
        qinit = robot.getConfig()
    if qdes is None:
        if isinstance(qinit,str):
            raise ValueError("Must specify one of qdes or qinit")
        qdes = qinit
        if len(qinit) != robot.numLinks():
            raise ValueError("Invalid size of initial configuration")
    if objective is None:
        objective = RobotConfigObjective(robot,qdes)

    if qinit == 'random' or qinit == 'random-collision-free':
        lower_bound = settings.minimum_constraint_value
        if qinit == 'random-collision-free':
            lower_bound = 0.0
        if qdes is None:
            raise ValueError("Can't specify no qdes with qinit = random")
        qinit = [0.0]*robot.numLinks()
        sampledFeasible = False
        dbest = -float('inf')
        qbest = qinit
        t0 = time.time()
        maxSamples = 50
        for sample in xrange(maxSamples):
            rad = float(sample)/(maxSamples-1)
            for i in xrange(len(qdes)):
                u = (random.uniform(-1,1)**2 + 1)*0.5
                vmin = qdes[i] + rad*(qmin[i]-qdes[i])
                vmax = qdes[i] + rad*(qmax[i]-qdes[i])
                qinit[i] = vmin + u*(vmax-vmin)
            gx = [float('inf')]*len(constraints)
            for i,c in enumerate(constraints):
                if isinstance(c,SemiInfiniteConstraintInterface):
                    gx[i] = c.eval_minimum(qinit)
                else:
                    gx[i] = c(qinit)
            dmin = min(gx)
            if dmin >= lower_bound:
                sampledFeasible = True
                break
            elif dmin > dbest:
                dbest = dmin
                qbest = qinit[:]
        if not sampledFeasible:
            if verbose >= 1: 
                print("optimizeCollFreeRobot: Could not generate an initial configuration that respects the minimum constraint value")
            return qbest,[qbest],[[] for c in constraints]
        t1 = time.time()
        print("Solved in time",t1-t0,"with distance to target",objective.value(qinit))

    qmin = np.asarray(qmin)
    qmax = np.asarray(qmax)
    res = optimizeSemiInfinite(objective,constraints,qinit,qmin,qmax,verbose=verbose,settings=settings)

    #format the output
    if not want_trace and not want_times and not want_constraint_pts:
        return res.x
    retlist = [res.x]
    if want_trace:
        retlist.append(res.trace)
    if want_times:
        retlist.append(res.trace_times)
    if want_constraints:
        retlist.append(res.instantiated_params)
    return retlist

def optimizeCollFreeTrajectory(trajcache,traj0,env,objective=None,constraints=None,greedyStart=False,
    settings=None,verbose=1,
    want_trace=True,want_times=True,want_constraints=True):
    """
    Optimizes a trajectory subject to collision-free constraints.

    Parameters:
    - trajcache: a RobotTrajectoryCache object dictating the form of the trajectory
    - traj0: an initial Trajectory or state
    - env: can be a RigidObject, TerrainModel, or geometry describing the static object.  It can also be a list of
      static objects.
    - objective: if given, an ObjectiveFunctionInterface that measures the trajectory cost.
    - constraints: if given, a list of constraints that overrides the default (collision free constraints
      between all robot links and environment objects).
    - greedyStart: if True, generates a new initial trajectory that tries to follow traj0 but obeys collision
      constraints.  This is done pointwise.
    - settings: a SemiInfiniteOptimizationSettings object to customize solver settings
    - verbose: controls how much output you want to see
    - want_trace,want_times,want_constraints: set to True if you want to return the
      trace, times, and/or constraints (see below)

    Returns:
        If want_trace, want_times, and want_constraints are all false, returns the resulting Trajectory traj.
        Otherwise, returns a tuple (traj,...) optionally containing:
        - trace: the list of Trajectory's at each major iteration
        - times: the list of major iteration times
        - constraints: a list containing the final set of active constraint points
    """
    if settings is None:
        settings = SemiInfiniteOptimizationSettings()
        settings.minimum_constraint_value = -0.02
    robot = trajcache.robot
    qmin,qmax = robot.getJointLimits()
    qmin = np.asarray(qmin)
    qmax = np.asarray(qmax)
    if isinstance(traj0,Trajectory):
        xinit = trajcache.trajectoryToState(traj0)
    else:
        xinit = traj0
    if greedyStart:
        assert env is not None,"GreedyStart requires the environment to be given"
        if verbose >= 1: print("Performing greedy start...")
        robot = trajcache.robot
        greedyconstraints,pairs = makeCollisionConstraints(trajcache.kinematics,env)
        traj0 = trajcache.stateToTrajectory(xinit)
        qlast = traj0.milestones[0]
        t0 = time.time()
        if trajcache.qstart is None: #free start point
            qobjective = RobotConfigObjective(robot,traj0.milestones[0])
            res = optimizeSemiInfinite(qobjective,greedyconstraints,qlast,qmin,qmax,verbose=0,settings=settings)
            if verbose >= 1: print("   Reduced start config f(x) from %g to %g"%(res.fx0,res.fx))
            if verbose >= 1: print("   Reduced start config g(x) from",res.gx0,"to",res.gx)
            traj0.milestones[0] = res.x
            qlast = res.x
        for i in xrange(1,len(traj0.milestones)-1):
            qobjective = RobotConfigObjective(robot,traj0.milestones[i])
            print("  Target",i,"=",traj0.milestones[i])
            res = optimizeSemiInfinite(qobjective,greedyconstraints,qlast,qmin,qmax,verbose=0,settings=settings)
            if verbose >= 1: print("   Reduced config",i,"f(x) from %g to %g"%(res.fx0,res.fx))
            if verbose >= 1: print("   Reduced config",i,"g(x) from",res.gx0,"to",res.gx)
            print("  Result",i,"=",res.x)
            traj0.milestones[i] = res.x
            qlast = res.x
        if trajcache.qend is None: #free end point
            qobjective = RobotConfigObjective(robot,traj0.milestones[-1])
            res = optimizeSemiInfinite(qobjective,greedyconstraints,qlast,qmin,qmax,verbose=0,settings=settings)
            if verbose >= 1: print("   Reduced last config f(x) from %g to %g"%(res.fx0,res.fx))
            if verbose >= 1: print("   Reduced last config g(x) from",res.gx0,"to",res.gx)
            traj0.milestones[-1] = res.x
            qlast = res.x
        t1 = time.time()

        xinit = trajcache.trajectoryToState(traj0)
        if verbose >= 1:
            objective = TrajectoryLengthObjective(trajcache)
            print("Completed in time %g, objective value %g, now proceeding to main optimization"%(t1-t0,objective.value(xinit)))
        #just return the initial trajectory
        if DEBUG_TRAJECTORY_INITIALIZATION:
            return traj0,[traj0],[[]]
        #raw_input()

    #proceed to main optimization
    xmin = np.hstack([qmin]*trajcache.numPoints)
    xmax = np.hstack([qmax]*trajcache.numPoints)
    if objective is None:
        objective = TrajectoryLengthObjective(trajcache)
    if constraints is None:
        if not hasattr(env,'__iter__'):
            env = [env]
        """
        constraints = []
        for e in env:
            if not isinstance(e,PenetrationDepthGeometry):
                e = PenetrationDepthGeometry(e.geometry())
            constraints += [RobotLinkTrajectoryCollisionConstraint(robot.link(i),e,trajcache) for i in xrange(robot.numLinks())]
        """
        for i,e in enumerate(env):
            if not isinstance(e,PenetrationDepthGeometry):
                env[i] = PenetrationDepthGeometry(e.geometry())
        constraints = [RobotTrajectoryCollisionConstraint(env,trajcache)]
    res = optimizeSemiInfinite(objective,constraints,xinit,xmin,xmax,verbose=verbose,settings=settings)
    
    #format the output
    traj = trajcache.stateToTrajectory(res.x)
    if not want_trace and not want_times and not want_constraint_pts:
        return traj
    retlist = [traj]
    if want_trace:
        retlist.append([trajcache.stateToTrajectory(v) for v in res.trace])
    if want_times:
        retlist.append(res.trace_times)
    if want_constraints:
        retlist.append(res.instantiated_params)
    return retlist
