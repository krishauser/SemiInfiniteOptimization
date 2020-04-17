#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 19:28:50 2020

@author: mengchao
"""

from klampt import *
from klampt.math import vectorops,so3,se3
from klampt.model.trajectory import Trajectory,SE3Trajectory
from klampt import vis
from klampt.io import resource
import numpy as np
from semiinfinite.geometryopt import *
from semiinfinite.sip import SemiInfiniteOptimizationSettings
import math
from cvxopt import matrix, div, max, min
from cvxopt.solvers import qp, lp
from numpy import linalg as LA

it_max = 10

#you can play with these parameters
gridres = 0.05
pcres = 0.02
resource.setDirectory('.')
obj1 = resource.get('data/sphere.off','Geometry3D')
obj2 = resource.get('data/sphere.off','Geometry3D')
obj3 = resource.get('data/sphere.off','Geometry3D')
obj4 = resource.get('data/sphere.off','Geometry3D')
obj5 = resource.get('data/sphere.off','Geometry3D')
obj6 = resource.get('data/sphere.off','Geometry3D')
obj7 = resource.get('data/sphere.off','Geometry3D')
obj8 = resource.get('data/sphere.off','Geometry3D')

geom1 = PenetrationDepthGeometry(obj1,gridres,pcres)
geom2 = PenetrationDepthGeometry(obj2,gridres,pcres)
geom3 = PenetrationDepthGeometry(obj3,gridres,pcres)
geom4 = PenetrationDepthGeometry(obj4,gridres,pcres)
geom5 = PenetrationDepthGeometry(obj5,gridres,pcres)
geom6 = PenetrationDepthGeometry(obj6,gridres,pcres)
geom7 = PenetrationDepthGeometry(obj7,gridres,pcres)
geom8 = PenetrationDepthGeometry(obj8,gridres,pcres)

#env = [geom1,geom2,geom3]
env = [geom1,geom2,geom3,geom5,geom6,geom7,geom8]
# Initial Transform for all the geometries
geom1.setTransform((so3.identity(),[0,0,0]))
#geom2.setTransform((so3.identity(),[2.4,0,0]))
#geom3.setTransform((so3.identity(),[1.2,2.07846,0]))
geom2.setTransform((so3.identity(),[2.4,0,1.5]))
geom3.setTransform((so3.identity(),[1.2,2.07846,1.5]))
#geom4.setTransform((so3.identity(),[1.2,0.69282,2]))
geom4.setTransform((so3.identity(),[5.2,0.4,5.2]))
geom5.setTransform((so3.identity(),[4.8,0,3]))
geom6.setTransform((so3.identity(),[3.6,2.07846,3]))
geom7.setTransform((so3.identity(),[7.2,0,4.5]))
geom8.setTransform((so3.identity(),[6,2.07846,4.5]))

# Used to plot the initial condition
vis.add("obj1",geom1.pc)
vis.add("obj2",geom2.pc)
vis.add("obj3",geom3.pc)
vis.add("obj4",geom4.grid)
vis.add("obj5",geom5.pc)
vis.add("obj6",geom6.pc)
vis.add("obj7",geom7.pc)
vis.add("obj8",geom8.pc)

vis.setColor("obj1",0,1,1,0.5)
vis.setColor("obj2",0,1,1,0.5)
vis.setColor("obj3",0,1,1,0.5)
vis.setColor("obj4",1,1,0,0.5)
vis.setColor("obj5",0,1,1,0.5)
vis.setColor("obj6",0,1,1,0.5)
vis.setColor("obj7",0,1,1,0.5)
vis.setColor("obj8",0,1,1,0.5)

vis.show()

# Parameters used in optimization
#I = []
#it = 0
#tau = 1
#sigma = 0.01
#delta = 0.1
#rou = 10
#epsilon = 5e-7
#presd = 1
#k = 0

it = 0
Alpha = []
VAR = []

while it < it_max: #and (it == 0 or (LA.norm(abs(matrix(Normal)*matrix(var[dim_1:dim_1+dim_2]) - matrix([0,0,9.8])),1) > 2 or (LA.norm(W*Y*e,1)) > 1)):# and converged_out == False:
        
    print 'it',it,'\n'     

    tau = 1
    sigma = 0.01
    delta = 0.1
    rou = 1
    epsilon = 5e-7
    presd = 1     
    # Establish collision constraints between the object and its environment
    cons = []
    I = []
    for geom in env:
        cons.append(ObjectCollisionConstraint(geom4,geom))
    for i,con in enumerate(cons):
        tmp = list(con.minvalue(geom4.getTransform())[0:2])
        tmp.append(i)
        tmp = tuple(tmp)
        I.append(tmp)
        
    var = geom4.getTransform()[1] + [0]*len(I) + [pair[0] for pair in I] 
    dim_0 = len(var)
    dim_1 = 3
    dim_2 = int((dim_0 - dim_1)/2)

    # Gradient of the objective function    
    df = matrix(np.zeros(dim_0),tc='d')
    df[dim_1 -1] = -1.0
    
    Normal = []
    for i in range(len(I)):
        Normal.append(vectorops.mul(env[I[i][2]].normal(I[i][1]),-1))
    Derivative = []
    for i in range(len(I)):
        Derivative.append(list(cons[I[i][2]].df_dx(geom4.getTransform(),I[i][1])[3:6]))
    
#    if it == 0:
#        plane1 = resource.get('data/plane.off','Geometry3D')
#        plane1_geom = PenetrationDepthGeometry(plane1,gridres,pcres)
#        plane1_geom.setTransform((so3.vector_rotation([0,0,1],Normal[0]),I[0][1]))
#        vis.add("plane1",plane1_geom.pc)
#        vis.setColor("plane1",1,1,0,0.5)
#        
#        plane2 = resource.get('data/plane.off','Geometry3D')
#        plane2_geom = PenetrationDepthGeometry(plane2,gridres,pcres)
#        plane2_geom.setTransform((so3.vector_rotation([0,0,1],Normal[1]),I[1][1]))
#        vis.add("plane2",plane2_geom.pc)
#        vis.setColor("plane2",1,1,0,0.5)
#        
#        plane3 = resource.get('data/plane.off','Geometry3D')
#        plane3_geom = PenetrationDepthGeometry(plane3,gridres,pcres)
#        plane3_geom.setTransform((so3.vector_rotation([0,0,1],Normal[2]),I[2][1]))
#        vis.add("plane3",plane3_geom.pc)
#        vis.setColor("plane3",1,1,0,0.5)
#        
#        plane4 = resource.get('data/plane.off','Geometry3D')
#        plane4_geom = PenetrationDepthGeometry(plane4,gridres,pcres)
#        plane4_geom.setTransform((so3.vector_rotation([0,0,1],Normal[3]),I[3][1]))
#        vis.add("plane4",plane4_geom.pc)
#        vis.setColor("plane4",1,1,0,0.5)
#        
#        plane5 = resource.get('data/plane.off','Geometry3D')
#        plane5_geom = PenetrationDepthGeometry(plane5,gridres,pcres)
#        plane5_geom.setTransform((so3.vector_rotation([0,0,1],Normal[4]),I[4][1]))
#        vis.add("plane5",plane5_geom.pc)
#        vis.setColor("plane5",1,1,0,0.5)
#        
#        plane6 = resource.get('data/plane.off','Geometry3D')
#        plane6_geom = PenetrationDepthGeometry(plane6,gridres,pcres)
#        plane6_geom.setTransform((so3.vector_rotation([0,0,1],Normal[5]),I[5][1]))
#        vis.add("plane6",plane6_geom.pc)
#        vis.setColor("plane6",1,1,0,0.5)
#    else:
#        plane1_geom.setTransform((so3.vector_rotation([0,0,1],Normal[0]),I[0][1]))
#        plane2_geom.setTransform((so3.vector_rotation([0,0,1],Normal[1]),I[1][1]))   
#        plane3_geom.setTransform((so3.vector_rotation([0,0,1],Normal[2]),I[2][1]))
#        plane4_geom.setTransform((so3.vector_rotation([0,0,1],Normal[3]),I[3][1]))
#        plane5_geom.setTransform((so3.vector_rotation([0,0,1],Normal[4]),I[4][1]))
#        plane6_geom.setTransform((so3.vector_rotation([0,0,1],Normal[5]),I[5][1]))
#    vis.remove("plane1")
#    vis.remove("plane2")
#    vis.remove("plane3")
#    vis.remove("plane4")
#    vis.remove("plane5")
#    plane4 = resource.get('data/plane.off','Geometry3D')
#    plane4_geom = PenetrationDepthGeometry(plane4,gridres,pcres)
#    plane4_geom.setTransform((so3.vector_rotation([0,0,1],Normal[3]),I[3][1]))
#    vis.add("plane4",plane4_geom.pc)
#    vis.setColor("plane4",1,1,0,0.5)
#    
#    plane5 = resource.get('data/plane.off','Geometry3D')
#    plane5_geom = PenetrationDepthGeometry(plane5,gridres,pcres)
#    plane5_geom.setTransform((so3.vector_rotation([0,0,1],Normal[4]),I[4][1]))
#    vis.add("plane5",plane5_geom.pc)
#    vis.setColor("plane5",1,1,0,0.5)
#    
#    plane6 = resource.get('data/plane.off','Geometry3D')
#    plane6_geom = PenetrationDepthGeometry(plane6,gridres,pcres)
#    plane6_geom.setTransform((so3.vector_rotation([0,0,1],Normal[5]),I[5][1]))
#    vis.add("plane6",plane6_geom.pc)
#    vis.setColor("plane6",1,1,0,0.5)
#    
#    plane7 = resource.get('data/plane.off','Geometry3D')
#    plane7_geom = PenetrationDepthGeometry(plane7,gridres,pcres)
#    plane7_geom.setTransform((so3.vector_rotation([0,0,1],Normal[6]),I[6][1]))
#    vis.add("plane7",plane7_geom.pc)
#    vis.setColor("plane7",1,1,0,0.5)
    
    # Inequality constarints
    G = np.zeros((dim_0 - dim_1,dim_0),'float') 
    for i in range(0,dim_0-dim_1):
        G[i][i+dim_1] = -1
    G = matrix(G,tc='d')
    
    h = np.zeros((dim_0 - dim_1,1),'float') 
    h = matrix(h,tc='d')
        
    # Equality constarints
    A = np.zeros((dim_1+dim_2,dim_0),'float')
    # Matrix B
    for i in range(0,dim_1):
        for j in range(dim_1,dim_1+dim_2):
            A[i][j] = Normal[j-dim_1][i]
    
    for i in range(dim_1,dim_1+dim_2):
        for j in range(0,dim_1):
            A[i][j] = Derivative[i-dim_1][j]
    
    for i in range(dim_1,dim_1+dim_2):
        for j in range(dim_1+dim_2,dim_0):
            if j-dim_1-dim_2 == i-dim_1:
                A[i][j] = -1
                    
#    for i in range(dim_1,dim_1+dim_2):
#        for j in range(0,dim_1):
#            A[i][j] = Normal[i-dim_1][j]
#    
#    for i in range(dim_1,dim_1+dim_2):
#        for j in range(dim_1+dim_2,dim_0):
#            if j-dim_1-dim_2 == i-dim_1:
#                A[i][j] = -1
    
    A = matrix(A,tc = 'd')
    
    b = matrix([0]*(dim_1+dim_2),tc = 'd')
    b[dim_1-1] = 9.8

    for i in range(dim_1,dim_1+dim_2):
        #b[i] = I[i-dim_1][1][0]*Normal[i-dim_1][0] + I[i-dim_1][1][1]*Normal[i-dim_1][1] + I[i-dim_1][1][2]*Normal[i-dim_1][2] + 1
        b[i] = - I[i-dim_1][0] + Derivative[i-dim_1][0] * var[0] + Derivative[i-dim_1][1] * var[1] + Derivative[i-dim_1][2] * var[2] 
        #b[i] = - I[i-dim_1][0] + Normal[i-dim_1][0] * var[0] + Normal[i-dim_1][1] * var[1] + Normal[i-dim_1][2] * var[2] - 1
        
#    P = np.zeros((dim_0,dim_0))
#    for i in range(dim_1,dim_0):
#        P[i][i] = 2
#    P = matrix(P,tc = 'd')
#    q = matrix(0.0,(dim_0,1))
    P = np.zeros((dim_0,dim_0))
    for i in range(0,dim_1):
        P[i][i] = 2
    for i in range(dim_1+dim_2,dim_0):
        P[i][i] = 1
    P = matrix(P,tc = 'd')
    q = matrix(0.0,(dim_0,1))
    for i in range(0,dim_1):
        q[i] = -2*var[i]
    result = qp(P=P,q=q,G=G,h=h,A=A,b=b,kktsolver='ldl',options={'kktreg':1e-6,'show_progress': False})
    #result = qp(P=P,q=q,G=G,h=h,A=A,b=b,kktsolver='ldl',options={'kktreg':1e-9,'show_progress': False})
    #result = qp(P=P,q=q,G=G,h=h,A=A,b=b,options={'show_progress': False})
    var = list(result['x'])
    print 'var initial', var, '\n'
        
    #print 'var_initial', var
    it_inner = 0 
    it_inner_max = 10
    converged_out = False
    
    while it_inner < it_inner_max and converged_out == False:
        print 'it_inner',it_inner,'\n'          
        # Step2 
        
        # Solve LP subproblem
        # Inequality constraints

        G = np.zeros((dim_2*4,dim_0+dim_2),'float') 
        for i in range (0,dim_2):
            for j in range(dim_1,dim_1+dim_2):
                if i == j-dim_1:
                    G[i][j] = -1.0 
        for i in range(dim_2, 2*dim_2):
            for j in range(dim_1+dim_2,dim_1 + 2*dim_2):
                if i- dim_2 == j - (dim_1+dim_2):
                    G[i][j] = -1.0 
        
        for i in range (2*dim_2,3*dim_2):
            for j in range(dim_1,dim_1+dim_2):
                if i - 2*dim_2 == j - dim_1:
                    G[i][j] = var[i - 2*dim_2 + dim_1 + dim_2]
    
        for i in range (2*dim_2,3*dim_2):
            for j in range(dim_1+dim_2,dim_1+2*dim_2):
                if i - 2*dim_2 == j - (dim_1+dim_2):
                    G[i][j] = var[i - 2*dim_2 + dim_1]
                    
        for i in range (2*dim_2,3*dim_2):
            for j in range(dim_1+2*dim_2,dim_1+3*dim_2):
                if i - 2*dim_2 == j-(dim_1+2*dim_2):
                    G[i][j] = -1.0
        
        for i in range (3*dim_2,4*dim_2):
            for j in range(dim_1+2*dim_2,dim_1+3*dim_2):
                if i - 3*dim_2 == j-(dim_1+2*dim_2):
                    G[i][j] = -1.0
            
        G = matrix(G,tc='d')
        
        h = np.zeros((dim_2*4,1),'float') 
        for i in range(0,2*dim_2):
            h[i] = var[i+dim_1]
        f_mat = np.diag(var[dim_1:dim_1+dim_2])
        d_mat = np.diag(var[dim_1+dim_2:dim_0])
        tmp = -(np.matmul(np.matmul(f_mat,d_mat),np.ones(dim_2)) - tau * np.ones(dim_2))
        for i in range(2*dim_2,3*dim_2):
            h[i] = tmp[i-2*dim_2]
        h = matrix(h,tc = 'd')
        
        # Equality constraints
        A = np.zeros((dim_1+dim_2,dim_0+dim_2),'float')
        # Matrix B
        for i in range(0,dim_1):
            for j in range(dim_1,dim_1+dim_2):
                A[i][j] = Normal[j-dim_1][i]
        # Matrix N
        for i in range(dim_1,dim_1+dim_2):
            for j in range(0,dim_1):
                A[i][j] = Derivative[i-dim_1][j]
        
        for i in range(dim_1,dim_1+dim_2):
            for j in range(dim_1+dim_2,dim_0):
                if j-dim_1-dim_2 == i-dim_1:
                    A[i][j] = -1
#        for i in range(dim_1,dim_1+dim_2):
#            for j in range(0,dim_1):
#                A[i][j] = Normal[i-dim_1][j]
#        
#        for i in range(dim_1,dim_1+dim_2):
#            for j in range(dim_1+dim_2,dim_0):
#                if j-dim_1-dim_2 == i-dim_1:
#                    A[i][j] = -1
                
        A = matrix(A,tc = 'd')
        
        b = matrix([0]*(dim_1+dim_2),tc = 'd')
        
        c = np.zeros(dim_0+dim_2) 
        for i in range(dim_1 + 2*dim_2,dim_0+dim_2):
            c[i] = 1
        c = matrix(c,tc = 'd')
        result = lp(c=c,G=G,h=h,A=A,b=b,kktsolver='ldl',options={'kktreg':1e-9,'show_progress': False})
        #result = lp(c=c,G=G,h=h,A=A,b=b,options={'show_progress': False})
        var_tilta = list(result['x'])
        
        # Solve modified QP problem
        # Inequality constraints     
        G = np.zeros((dim_2*3,dim_0),'float') 
        for i in range (0,dim_2):
            for j in range(dim_1,dim_1+dim_2):
                if i == j-dim_1:
                    G[i][j] = -1.0 
        for i in range(dim_2,2*dim_2):
            for j in range(dim_1+dim_2,dim_1+2*dim_2):
                if i- dim_2 == j - (dim_1+dim_2):
                    G[i][j] = -1.0 
        
        for i in range (2*dim_2,3*dim_2):
            for j in range(dim_1,dim_1+dim_2):
                if i - 2*dim_2 == j - dim_1:
                    G[i][j] = var[i - 2*dim_2 + dim_1 + dim_2]
    
        for i in range (2*dim_2,3*dim_2):
            for j in range(dim_1+dim_2,dim_1+2*dim_2):
                if i - 2*dim_2 == j - (dim_1+dim_2):
                    G[i][j] = var[i - 2*dim_2 + dim_1]
            
        G = matrix(G,tc='d')
        
        h = np.zeros((dim_2*3,1),'float') 
        for i in range(0,2*dim_2):
            h[i] = var[i+dim_1]
        f_mat = np.diag(var[dim_1:dim_1+dim_2])
        d_mat = np.diag(var[dim_1+dim_2:dim_0])
        df_mat = np.diag(var_tilta[dim_1:dim_1+dim_2])
        dd_mat = np.diag(var_tilta[dim_1+dim_2:dim_0])
        tmp1 = -(np.matmul(np.matmul(f_mat,d_mat),np.ones(dim_2)) - tau * np.ones(dim_2))
        tmp2 = np.matmul(np.matmul(f_mat,dd_mat),np.ones(dim_2)) + np.matmul(np.matmul(d_mat,df_mat),np.ones(dim_2))
        for i in range(2*dim_2,3*dim_2):
            h[i] = max(tmp1[i-2*dim_2],tmp2[i-2*dim_2])
            #h[i] = max(tmp1[i-2*dim_2])
        h = matrix(h,tc = 'd')
        
        # Equality constraints
        A = np.zeros((dim_1+dim_2,dim_0),'float')
        # Matrix B
        for i in range(0,dim_1):
            for j in range(dim_1,dim_1+dim_2):
                A[i][j] = Normal[j-dim_1][i]
        # Matrix N
        for i in range(dim_1,dim_1+dim_2):
            for j in range(0,dim_1):
                A[i][j] = Derivative[i-dim_1][j]
        for i in range(dim_1,dim_1+dim_2):
            for j in range(dim_1+dim_2,dim_0):
                if j-dim_1-dim_2 == i-dim_1:
                    A[i][j] = -1
#        for i in range(dim_1,dim_1+dim_2):
#            for j in range(0,dim_1):
#                A[i][j] = Normal[i-dim_1][j]
#        
#        for i in range(dim_1,dim_1+dim_2):
#            for j in range(dim_1+dim_2,dim_0):
#                if j-dim_1-dim_2 == i-dim_1:
#                    A[i][j] = -1
        A = matrix(A,tc = 'd')
        
        b = matrix([0]*(dim_1+dim_2),tc = 'd')
        
        H = 2*matrix(np.identity(dim_0),tc='d')
        #H = matrix(np.zeros((dim_0,dim_0)),tc='d')
        result = qp(P=H,q=df,G=G,h=h,A=A,b=b,kktsolver='ldl', options={'kktreg':1e-9,'show_progress': False})
        #result = qp(P=H,q=df,G=G,h=h,A=A,b=b,options={'show_progress': False})
        d = list(result['x'])
        
        print 'd',matrix(d),'\n'
        W = matrix(np.diag(var[dim_1+dim_2:dim_0]),tc='d')
        Y = matrix(np.diag(var[dim_1:dim_1+dim_2]),tc='d')
        v = matrix([var_tilta[dim_0:dim_0+dim_2]],tc='d')
        e = matrix(np.ones(dim_2),tc='d')
        d_mat = matrix(d)
        
        flag = 0
        
        # Step 3
        if (LA.norm(d,2) <= epsilon or ((df.T * d_mat)[0] >= -0.1*epsilon)) \
           and LA.norm(max(W*Y*e - tau*e,0),1) <= epsilon:
            var = var
            rou = rou
            presd = 0
            flag = 1
        if (df.T*d_mat + 0.5 * d_mat.T * H * d_mat \
             - rou * (LA.norm(max(W*Y*e - tau*e,0),1) - LA.norm(v,1)))[0] <= 0:
            rou = rou
        else:
            rou = max(2*rou,((df.T*d_mat + 0.5 * d_mat.T * H * d_mat)/(LA.norm(max(W*Y*e - tau*e,0),1) - LA.norm(v,1)))[0])

        # Step 4
        if flag == 0:
            Delta = (df.T*d_mat - rou * (LA.norm(max(W*Y*e - tau*e,0),1)) - LA.norm(v,1))[0]
            converged = False
            it_max_LS = 20
            iteration_LS = 0
            alpha = 1
            phi_0 = -var[2] + rou * (LA.norm(max(W*Y*e-tau*e,0),1))
            #phi_0 = rou * (LA.norm(W*Y*e - tau*e,1))
            print 'phi_0', phi_0, '\n'
            while (not converged) and (iteration_LS < it_max_LS):
                var_tmp = (np.array(var) + alpha * np.array(d)).tolist()
                W_tmp = matrix(np.diag(var_tmp[dim_1+dim_2:dim_0]),tc='d')
                Y_tmp = matrix(np.diag(var_tmp[dim_1:dim_1+dim_2]),tc='d')
                phi = -var_tmp[2] + rou * (LA.norm(max(W_tmp*Y_tmp*e-tau*e,0),1))
                #phi = rou * (LA.norm(max(W_tmp*Y_tmp*e - tau*e,0),1))
                if not phi <= (phi_0 + sigma * alpha * Delta):
                    alpha = alpha * 0.6
                    iteration_LS += 1
                else:
                    converged = True
                    Alpha.append(alpha)
                    var = (np.array(var) + alpha * np.array(d)).tolist()
                    VAR.append(var)
                print 'alpha',alpha,'var',var[2],'phi',phi
        # Step 5
        if (tau <= epsilon and (presd == 0 or LA.norm(W*Y*e - tau*e,1) - LA.norm(v,1) <= 1e-6 * epsilon)) \
             or (LA.norm(W*Y*e,np.inf) <= epsilon and presd == 0):
            converged_out = True
        
        if tau > epsilon:
            tau = delta * tau
        else:
            tau = tau  
        
        it_inner += 1
    geom4.setTransform((so3.identity(),var[0:3]))
    it += 1

