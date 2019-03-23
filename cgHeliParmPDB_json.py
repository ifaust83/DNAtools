#!/usr/bin/python
import numpy as np
from numpy import linalg as LA
import argparse
from argparse import RawTextHelpFormatter
import MDAnalysis
import sys
import time
import json 
import pandas as pd 
import os

# Locate the directory in which the script is stored.
# The __file__ variable contains the path to the python file,
# we save the directory part of that path. This is useful latter on to locate
# the data library.
SCRIPT_DIRECTORY = os.path.dirname(__file__)

start_time = time.time()
parser = argparse.ArgumentParser(description='Calculate helical parameters for dsDNA')
parser.add_argument('-i', dest="pdb", metavar='', 
    required=True, help='Input file (PDB)')
parser.add_argument('-o', dest="outfile", metavar='', required=True,
        help='Output filename')
args = parser.parse_args()
    
def lsfit(residue_name,residue_number):
    base = mol.residues.residues[residue_number]
    if 'A' in residue_name:
        xprm = base.atoms.select_atoms("name SC1 or name SC2 or name SC3 or name SC4")
        stnd = MDAnalysis.Universe(library_path + 'CG_A_std.pdb')
    elif 'C' in residue_name:
        xprm = base.atoms.select_atoms("name SC1 or name SC2 or name SC3")
        stnd = MDAnalysis.Universe(library_path + 'CG_C_std.pdb')
    elif 'G' in residue_name: 
        xprm = base.atoms.select_atoms("name SC1 or name SC2 or name SC3 or name SC4")
        stnd = MDAnalysis.Universe(library_path + 'CG_G_std.pdb')
    elif 'T' in residue_name:    
        xprm = base.atoms.select_atoms("name SC1 or name SC2 or name SC3")
        stnd = MDAnalysis.Universe(library_path + 'CG_T_std.pdb')
    stnd_cog = stnd.atoms.centroid()
    xprm_cog = xprm.atoms.centroid()
    S = stnd.atoms.positions
    E = xprm.atoms.positions
    # Covariance matrix (C) calculation.
    bal = np.dot(S.transpose(),E)
    i = np.ones(S.shape[0], dtype=float).reshape(S.shape[0],1)
    lab1 = np.dot(S.transpose(),i)
    lab2 = np.dot(i.transpose(),E)
    C =  ( bal - np.dot(lab1,lab2) / S.shape[0] )/ (E.shape[0] - 1)
    # From C we generate the 4x4 symmetric matrix (M). 
    M = np.array([[C[0,0]+C[1,1]+C[2,2],C[1,2]-C[2,1],
                    C[2,0]-C[0,2],C[0,1]-C[1,0]],
                    [C[1,2]-C[2,1],C[0,0]-C[1,1]-C[2,2],C[0,1]+C[1,0],
                    C[2,0]+C[0,2]],
                    [C[2,0]-C[0,2],C[0,1]+C[1,0],-C[0,0]+C[1,1]-C[2,2],
                    C[1,2]+C[2,1]],
                    [C[0,1]-C[1,0],C[2,0]+C[0,2],C[1,2]+C[2,1],
                    -C[0,0]-C[1,1]+C[2,2]]])
    # Eigenvalue (w) and eigenvector (v) of M and eigenvector (q) associated 
    #     to the highest eigenvalue.
    w, v = np.linalg.eig(M)
    q = -v[:,w.argmax()]
    # Origin (o) and orientation matrix (R) of the base define the BASE 
    #     reference frame.
    R = np.array([[q[0]*q[0]+q[1]*q[1]-q[2]*q[2]-q[3]*q[3],
                    2*(q[1]*q[2]-q[0]*q[3]),
                    2*(q[1]*q[3]+q[0]*q[2])],
                    [2*(q[2]*q[1]+q[0]*q[3]),
                    q[0]*q[0]-q[1]*q[1]+q[2]*q[2]-q[3]*q[3],
                    2*(q[2]*q[3]-q[0]*q[1])],
                    [2*(q[3]*q[1]-q[0]*q[2]),
                    2*(q[3]*q[2]+q[0]*q[1]),
                    q[0]*q[0]-q[1]*q[1]-q[2]*q[2]+q[3]*q[3]]])
    o = xprm_cog - np.dot(stnd_cog,R.transpose())
    return o, R.reshape(1,9)


def bpFrame(a,b):
    oribp = np.add(base_ori[base_a,:],base_ori[base_b,:])/2
    matbp = np.add(base_orient[base_a,:].reshape(3,3),\
            base_orient[base_b,:].reshape(3,3))/2
    return oribp,matbp.reshape(1,9)


def rotMatrix(v,ang):
    matrix = np.array([[np.cos(ang)+(1-np.cos(ang))*v[0],
                        (1-np.cos(ang))*v[0]*v[1]-v[2]*np.sin(ang),
                        (1-np.cos(ang))*v[0]*v[1]+v[1]*np.sin(ang)],
                        [(1-np.cos(ang))*v[0]*v[1]+v[2]*np.sin(ang),
                        np.cos(ang)+(1-np.cos(ang))*v[1]*v[1],
                        (1-np.cos(ang))*v[1]*v[2]-v[0]*np.sin(ang)],
                        [(1-np.cos(ang))*v[0]*v[2]-v[1]*np.sin(ang),
                        (1-np.cos(ang))*v[1]*v[2]+v[0]*np.sin(ang),
                        np.cos(ang)+(1-np.cos(ang))*v[2]*v[2]]])
    return matrix
    

def angle(v1,v2):
    return np.arccos(np.dot(v1,v2) / (LA.norm(v1) * LA.norm(v2)))


def bpParam(o1,R1,o2,R2):
    # Hinge axis (hinge) calculation.
    a1 = np.cross(R2[:,1],R1[:,1])
    gamma = a1/np.linalg.norm(a1)    
    # the RollTilt angle (RollTiltAngle) is the net bending angle.
    BuckleOpeningAngle = np.degrees(angle(R2[:,1],R1[:,1]))
    R1p = np.dot(rotMatrix(gamma,np.radians(-BuckleOpeningAngle/2)),R1)
    R2p = np.dot(rotMatrix(gamma,np.radians(BuckleOpeningAngle/2)),R2)
    Rm = np.add(R1p,R2p)/2
    om = np.add(o1,o2)/2
    propeller = np.degrees(angle(R2p[:,0],R1p[:,0]))
    propeller_sign = np.dot(np.cross(R2p[:,0],R1p[:,0]),Rm[:,1])
    if propeller_sign < 0:
        propeller = -propeller
    phase = np.degrees(angle(gamma,Rm[:,0]))
    phase_sign = np.dot(np.cross(gamma,Rm[:,0]),Rm[:,1])
    if phase_sign <0:
        phase = -phase
    buckle = BuckleOpeningAngle*np.cos(np.radians(phase))
    opening = BuckleOpeningAngle*np.sin(np.radians(phase))
    shear, stretch, stagger = np.dot(o1-o2,Rm)
    
    return om, Rm.reshape(1,9), shear, stretch, stagger, buckle, propeller, opening

def localHeliParam(o1,R1,o2,R2):
    # Hinge axis (hinge) calculation.
    a1 = np.cross(R2[:,0]-R1[:,0],R2[:,1]-R1[:,1])
    localhelicalaxis = a1/np.linalg.norm(a1)    
    # Calculating the TipIncl angle (TipInclAngle).
    TipInclAngle = np.degrees(angle(localhelicalaxis,R1[:,2]))
    hinge_axis = np.cross(localhelicalaxis,R1[:,2])
    norm_hinge_axis = hinge_axis/np.linalg.norm(hinge_axis)
    H1 = np.dot(rotMatrix(norm_hinge_axis,np.radians(-TipInclAngle)),R1)
    TipInclAngle2 = np.degrees(angle(localhelicalaxis,R2[:,2]))
    hinge_axis2 = np.cross(localhelicalaxis,R2[:,2])
    norm_hinge_axis2 = hinge_axis2/np.linalg.norm(hinge_axis2)
    H2 = np.dot(rotMatrix(norm_hinge_axis2,np.radians(-TipInclAngle)),R2)
    Hm = np.add(H1,H2)/2
    htwist = np.degrees(angle(H1[:,1],H2[:,1]))
    hrise = np.dot(o2-o1,localhelicalaxis)
    phase_angle = np.degrees(angle(norm_hinge_axis,H1[:,1]))
    phase_sign = np.dot(np.cross(norm_hinge_axis,Hm[:,1]),localhelicalaxis)
    if phase_sign < 0:
        phase_angle = -phase_angle
    tip = TipInclAngle*np.cos(np.radians(phase_angle))
    incl = TipInclAngle*np.sin(np.radians(phase_angle))
    AB = (o2-o1) - hrise*localhelicalaxis
    AB = np.matrix(AB)
    angleAD = 90 - htwist*0.5
    AD = np.dot(rotMatrix(localhelicalaxis,np.radians(angleAD)),AB.transpose())
    norm_AD = AD/np.linalg.norm(AD)
    norm_AD = norm_AD.transpose()
    magnitude_AD = np.linalg.norm(AB)*0.5/np.sin(np.radians(0.5*htwist))
    o1_h = o1 + magnitude_AD*norm_AD
    o2_h = o1_h + hrise*localhelicalaxis
    xdisp = np.dot(o1-o1_h,H1[:,0])
    ydisp = np.dot(o1-o1_h,H1[:,1])

    return xdisp, ydisp, hrise, incl, tip, htwist

def stepFrame(o1,R1,o2,R2):
    # Hinge axis (hinge) calculation.
    a1 = np.cross(R1[:,2],R2[:,2])
    hinge_axis = a1/np.linalg.norm(a1)    
    # the RollTilt angle (RollTiltAngle) is the net bending angle.
    RollTiltAngle = np.degrees(angle(R1[:,2],R2[:,2]))
    R1p = np.dot(rotMatrix(hinge_axis,np.radians(RollTiltAngle/2)),R1)
    R2p = np.dot(rotMatrix(hinge_axis,np.radians(-RollTiltAngle/2)),R2)
    Rm = np.add(R1p,R2p)/2
    om = np.add(o1,o2)/2
    shift, slide, rise = np.dot(o2-o1,Rm)
    twist = np.degrees(angle(R1[:,1],R2[:,1]))
    twist_sign = np.dot(np.cross(R1p[:,1],R2p[:,1]),Rm[:,2])
    if twist_sign < 0:
        twist = -twist
    phase = np.degrees(angle(hinge_axis,Rm[:,1]))
    phase_sign = np.dot(np.cross(hinge_axis,Rm[:,1]),Rm[:,2])
    if phase_sign <0:
        phase = -phase
    roll = RollTiltAngle*np.cos(np.radians(phase))
    tilt = RollTiltAngle*np.sin(np.radians(phase))
#    print shift, slide, rise, tilt, roll, twist
    
    return om, Rm.reshape(1,9), shift, slide, rise, tilt, roll, twist


def writeCenterCoords(centers,filename):
    ''' Saves the coordinates of the centers to a gro file.
    '''
    pdbfile = open(args.outfile+"_"+filename+'.pdb', 'w')
    pdbfile.write('Coordinates for all bases origins\n')
    points = len(centers)
    pdbfile.write(' {}\n'.format(points))
    for i in range(points):
	atom = i+1
        pdbfile.write('ATOM{:7d} 1CE   CEN  {:3d}    {:8.3f}{:8.3f}{:8.3f}\n'.format(
            atom, atom, centers[i,0], centers[i,1], centers[i,2]))
#    pdbfile.write('   100.000 100.000 100.000\n')
    pdbfile.close()
    
        
def write2json(data,file):
    f = open(args.outfile+"_"+file+".json", "w+b")
    f.write(data+'\n')
    
# input files
#ifile = raw_input('PDB file: ')
library_path = os.path.join(SCRIPT_DIRECTORY, "data")
mol = MDAnalysis.Universe(args.pdb)
total_res = len(mol.residues.resids)
basepairs = total_res/2
steps = (total_res/2)-1
#########################
# Bases frame
#########################
# loop through each residue (base) to calculate their corresponding base reference frame
base_ori,base_orient = np.zeros((total_res,3)),np.zeros((total_res,9))
basepair_parameters = np.zeros((basepairs,6))
local_parameters = np.zeros((steps,6))
step_parameters = np.zeros((steps,6))
for i in mol.residues.resids:
    res = i-1
    ka,la = lsfit(mol.residues.resnames[res],res)
    base_ori[res,:] = ka
    base_orient[res,:] = la
    # reversing y- and z-axes to made z-axis parallel with complementary base.
    if res >= total_res/2:
        base_mat = base_orient[res,:].reshape(3,3)
        base_mat[:,1],base_mat[:,2] = -1*base_mat[:,1],-1*base_mat[:,2]
        
        
#writeCenterCoords(base_ori,"nofit")
# writeCenterCoords(base_ori,"fitok")

#########################
# Base pair frame
#########################
bp_ori,bp_orient = np.zeros((total_res/2,3)),np.zeros((total_res/2,9))
for i in range(total_res/2):
    base_a = i
    base_b = total_res-i-1
    jo, ck, she, str, sta, buc, pro, ope = bpParam(base_ori[base_a,:],base_orient[base_a,:].reshape(3,3),base_ori[base_b,:],base_orient[base_b,:].reshape(3,3))
    basepair_parameters[i,:] = she, str, sta, buc, pro, ope
    ev, na = bpFrame(base_a,base_b)
    bp_ori[i,:] = ev
    bp_orient[i,:] = na
    
# writeCenterCoords(bp_ori,"bp")


#########################
# Base pair step frame
#########################
step_ori,step_orient = np.zeros((steps,3)),np.zeros((steps,9))
for i in range(steps):
    bp_a = i
    bp_b = i+1
    o_bpa = bp_ori[bp_a,:]
    R_bpa = bp_orient[bp_a,:].reshape(3,3)
    o_bpb = bp_ori[bp_b,:]
    R_bpb = bp_orient[bp_b,:].reshape(3,3)
    xdsp, ydsp, hrs, ncl, tp, htw = localHeliParam(o_bpa,R_bpa,o_bpb,R_bpb)
    local_parameters[i,:] = xdsp, ydsp, hrs, ncl, tp, htw
    ja, cl, sh, sl, ri, ti, ro, tw = stepFrame(o_bpa,R_bpa,o_bpb,R_bpb)
    step_ori[i,:] = ja
    step_orient[i,:] = cl
    step_parameters[i,:] = sh, sl, ri, ti, ro, tw

# writeCenterCoords(step_ori,"midframe") 
df_bpParam = pd.DataFrame(basepair_parameters, columns=['Shear','Stretch','Stagger','Buckle','Propeller','Opening'])
bpParam = df_bpParam.to_json(orient='columns')
df_stepParam = pd.DataFrame(step_parameters, columns=['Shift','Slide','Rise','Tilt','Roll','Twist'])
stepParam = df_stepParam.to_json(orient='columns')
df_localParam = pd.DataFrame(local_parameters, columns=['X-disp','Y-disp','H-rise','Inclination','Tip','H-twist'])
localParam = df_localParam.to_json(orient='columns')

write2json(bpParam, "basepair")
write2json(stepParam, "step")
write2json(localParam, "local")

print("--- %s seconds ---" % (time.time() - start_time))
