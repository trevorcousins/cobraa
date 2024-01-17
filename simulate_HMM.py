from transition_matrix import *
from BaumWelch import *
from utils import *
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import matrix_power
from numpy import linalg as LA
from numba import njit, jit
import pdb
import sys
import time
import psutil
import argparse
from scipy.optimize import minimize, minimize_scalar, LinearConstraint
from datetime import datetime
from joblib import Parallel, delayed
from scipy import linalg

def get_stationary_distribution_theory(matrix):
    # given theoretical matrix, calculate stationary distribution of markov chain
    eigvals, eigvecs = linalg.eig(matrix, left=True, right=False)
    theory_stat_dist = eigvecs[:,0]/eigvecs[:,0].sum()
    return theory_stat_dist

def generate_seq(L,init_dist,Q,E,D):
#     pdb.set_trace()
    # generate sequence with hidden states and emissions
    # returns index of state and emission, corresponding to states_str and emissions_str
    len_emissions = len(E[0,:]) # number of emissions
    sequence = np.zeros(shape=(2,L),dtype=int) # 2 rows, one for (hidden) state, one for emission
    state = np.random.choice(range(D),1,p=init_dist)[0]
    emiss = np.random.choice(range(len_emissions),1,p=E[state,:])[0]
    sequence[:,0] = [state,emiss]
#     pdb.set_trace()
    for i in range(L-1):
        state = np.random.choice(D,1,p=Q[state,:])[0]
        emiss = np.random.choice(range(len_emissions),1,p=E[state,:])[0]
        sequence[:,i+1] = [state,emiss]
    return sequence

def parse_lambda(lambda_string):
    lambda_list=lambda_string.split(',')
    lambda_list=[float(i) for i in lambda_list]
    lambda_values = np.array(lambda_list)
    return lambda_values

def write_mhs(pos,filename,chrom):
    # pos is index of hets
    # chrom is int
    current_chr = f'chr{chrom}'
    diff_pos = pos[1:] - pos[0:-1]
    SSPSS = np.concatenate(([pos[0]] ,diff_pos))
    gt = ['01']*len(pos)
    chr_label = [current_chr]*len(pos)

    with open(filename,'w') as f:
        lis=[chr_label,pos,SSPSS,gt]
        for x in zip(*lis):
            f.write("{0}\t{1}\t{2}\t{3}\n".format(*x))
    print(f'\twritten mhs file to {filename}')
    return None

def write_coal_data(sequence,changepoints,bin_size,T,filename):
    zcoal_data = np.zeros(shape=(2,len(changepoints)+1))
    zcoal_data[1,0] = sequence[0,0]
    zcoal_data[1,1:] = sequence[0,changepoints]
    zcoal_data[0,0] = 0
    zcoal_data[0,1:] = changepoints*bin_size
    # np.vstack([coal_data,np.array([seqlen,sequence[0,-1]])])
    zcoal_data = zcoal_data.T
    starts = zcoal_data[:,0]
    ends = np.concatenate([zcoal_data[1:,0],[seqlen]])
    coal_index =  zcoal_data[:,1]
    coal_data = np.zeros(shape=(len(ends),3),dtype=int)
    coal_data[:,0] = starts
    coal_data[:,1] = ends
    coal_data[:,2] = coal_index
    zz1 = ",".join([str(i) for i in T])
    zz2 = f'Time boundaries (coalescent units) = {zz1}\n'
    zz3 = 'start stop coalescent_index'
    np.savetxt(filename,coal_data,header=f'{zz2+zz3}',fmt="%s")

    print(f'\twritten coaldata to {filename}')
    return None


# def get_coal_data(seqlen,sequence):
    
#     change_points = []
#     change_points.append([0,sequence[1,0]])
#     for i in range(1,seq_length):
#         if sequence[1,i] != sequence[1,i-1]:
#         change_points.append([i,sequence[1,i]])
#     return change_points



# parse args

parser = argparse.ArgumentParser(description="Inputs and parameters")

parser.add_argument('-L','--seqlen',help='Length of simulation',required=True,type=float)
parser.add_argument('-D','--D',help='The number of time windows to use in inference',required=True,type=int)
parser.add_argument('-spread_1','--spread_1',help='Parameter controlling the time interval boundaries',required=False,type=float,default=0.1)
parser.add_argument('-spread_2','--spread_2',help='Parameter controlling the time interval boundaries',required=False,type=float,default=50)
parser.add_argument('-bin_size','--bin_size',help='Adjust recombination rate to bin this many bases together', required=False,type=int,default=1)
parser.add_argument('-rho','--rho',help='The scaled recombination rate; if p is per gen per bp recombination rate, and 2N is the diploid effective size, rho =4Np',required=True,type=float)
parser.add_argument('-theta','--theta',help='The scaled mutation rate; if mu is per gen per bp mutation rate, and 2N is the diploid effective size, theta =4Nmu',required=True,type=float)
parser.add_argument('-ts','--ts',help='Index of T_S, must be less than D (can be "None" for panmixia)',required=False,default=None,type=int)
parser.add_argument('-te','--te',help='Index of T_E, must be less than D (can be "None" for panmixia)',required=False,default=None,type=int)
parser.add_argument('-gamma','--gamma',help='gamma',nargs='?',const=False,type=float)
parser.add_argument('-lambda_A','--lambda_A',help='inverse pop sizes for A',required=False,type=str)
parser.add_argument('-lambda_B','--lambda_B',help='inverse pop sizes for B',nargs='?',const=False,type=str)
parser.add_argument('-midpoint_transitions','--midpoint_transitions',help='Whether to take midpoint in transitions',type=str, required=False,default="False")
parser.add_argument('-midpoint_emissions','--midpoint_emissions',help='Whether to take midpoint for the final two boundaries in the emission probabilities (take the midpoint at all other boundaries by default)',type=str, required=False,default="False")
parser.add_argument('-final_T_factor','--final_T_factor',help='If given, for the final time boundary take T[-2]*factor. Otherwise write according to sequence',type=str, required=False,default="False")
parser.add_argument('-recombnoexp','--recombnoexp',help='Model for recombination probability; either exponential (approximation with Taylor series) or standard',default=False,action='store_true')
parser.add_argument('-o_mhs','--o_mhs',help='Output path for mhsfile',required=False,type=str)
parser.add_argument('-o_coal','--o_coal',help='Output path for coal data',required=False,type=str)


args = parser.parse_args()
zargs = dir(args)
zargs = [zarg for zarg in zargs if zarg[0]!='_']
for zarg in zargs:
    print(f'{zarg} is ',end='')
    exec(f'{zarg}=args.{zarg}')
    exec(f'print(args.{zarg})')
seqlen = int(seqlen)

if lambda_A==None:
    lambda_A = np.ones(D)
else:
    lambda_A = parse_lambda(lambda_A)
    if len(lambda_A)!=D:
        print(f'length of lambda_A={len(lambda_A)} is not equal to D = {D}. Aborting')
        sys.exit()
    
if gamma==None and ts==None and te==None:
    print(f'panmictic simulation')
elif gamma!=None and ts!=None and te!=None:
    print(f'structured simulation')
    if lambda_B==None:
        lambda_B = np.ones(D)
    else:
        lambda_B = parse_lambda(lambda_B)
        if len(lambda_B)!=D:
            print(f'length of lambda_B={len(lambda_B)} is not equal to D = {D}. Aborting')
            sys.exit()
    if te<=ts:
        print(f'ts={ts} and te={te}; but ts must be smaller than te. Aborting')
        sys.exit()
else:
    print(f'Input parameters not valid. Aborting.')
    sys.exit()
if o_coal==None:
    now = datetime.now()
    dt_string = now.strftime("%Y%m%d_%H%M%S")
    o_coal = f'{os.getcwd()}/{dt_string}_coaldata.txt.gz'
if o_mhs==None:    
    now = datetime.now()
    dt_string = now.strftime("%Y%m%d_%H%M%S")
    o_mhs = f'{os.getcwd()}/{dt_string}.mhs'
tm = Transition_Matrix(D=D,spread_1=spread_1,spread_2=spread_2,midpoint_transitions=midpoint_transitions) 
T = tm.T
jmax = 50
Q = tm.write_tm(lambda_A=lambda_A,lambda_B=lambda_B,T_S_index=ts,T_E_index=te,gamma=gamma,check=True,rho=rho,exponential=not recombnoexp) # initialise transition matrix object
E = write_emission_probs(D,bin_size,theta,jmax,T)
# init_dist = get_stationary_distribution_theory(Q) # TODO Fix this
init_dist = np.ones(D)/D
sequence = generate_seq(seqlen,init_dist,Q,E,D)
# change_points = get_coal_data(seqlen,sequence)
changepoints = np.where(sequence[0,1:]-sequence[0,0:-1]!=0)[0]
changepoints+=1
write_mhs(np.where(sequence[1,:]==1)[0],o_mhs,'SIM')
write_coal_data(sequence,changepoints,bin_size,T,o_coal)

    

