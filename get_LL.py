# Given a simulated Markov chain of some model, calculate the LL under the true model and compare this to the competing model with matching coalescent rate
# E.g. simulate a structured model; record LL under structured model, then record LL under panmictic model with same coalescence rate
    # Then do the reverse (sim panmictic model; record LL under panmictic model, then record LL under structured model with same coalescence rate)
# Evaluate LL under what grid???

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
    # generate sequence with hidden states and emissions
    # returns index of state and emission, corresponding to states_str and emissions_str
    len_emissions = len(E[0,:]) # number of emissions
    sequence = np.zeros(shape=(2,L),dtype=int) # 2 rows, one for (hidden) state, one for emission
    state = np.random.choice(range(D),1,p=init_dist)[0]
    emiss = np.random.choice(range(len_emissions),1,p=E[state,:])[0]
    sequence[:,0] = [state,emiss]
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

@njit
def markov_LL(x,Q):
    # X is sequence of Markov chain
    # Q is transition matrix
    # init_dist is the initial distribution
    transitions = np.ones(len(x)-1)
    for i in range(0,len(x)-1):
        transitions[i] = Q[x[i],x[i+1]]
    LL = np.sum(np.log(transitions))
    return LL

# def get_coal_data(seqlen,sequence):
    
#     change_points = []
#     change_points.append([0,sequence[1,0]])
#     for i in range(1,seq_length):
#         if sequence[1,i] != sequence[1,i-1]:
#         change_points.append([i,sequence[1,i]])
#     return change_points



# parse args

parser = argparse.ArgumentParser(description="Inputs and parameters")
parser.add_argument('-i','--i',help='Markov LL: sequence of coalescent information. HMM LL: mhs file',required=True,type=str)
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
parser.add_argument('-HMM_LL','--HMM_LL',help='Get LL from full HMM (as oppposed to Markov sequence)',action='store_true')
parser.add_argument('-WETM','--WETM',help='If doing the LL of a HMM (i.e. argument -HMM_LL given), write the expected transition matrix to this file',default='',type=str)

args = parser.parse_args()
zargs = dir(args)
zargs = [zarg for zarg in zargs if zarg[0]!='_']
for zarg in zargs:
    print(f'{zarg} is ',end='')
    exec(f'{zarg}=args.{zarg}')
    exec(f'print(args.{zarg})')

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




tm = Transition_Matrix(D=D,spread_1=spread_1,spread_2=spread_2,midpoint_transitions=midpoint_transitions) 
T = tm.T
jmax = 50
Q = tm.write_tm(lambda_A=lambda_A,lambda_B=lambda_B,T_S_index=ts,T_E_index=te,gamma=gamma,check=True,rho=rho*bin_size,exponential=not recombnoexp) # initialise transition matrix object
E = write_emission_probs(D,bin_size,theta,jmax,T)
# init_dist = get_stationary_distribution_theory(Q) # TODO Fix this
init_dist = np.ones(D)/D

if HMM_LL:
    Q_array = np.zeros(shape=(1,D,D))
    Q_array[0,:,:] = Q
    # sequences_info = Parallel(n_jobs=cores, backend='loky')(delayed(bin_sequence)(in_path,bin_size,mhs_files_M_file,mhs_files_R_file) for in_path in files_paths) # returns for mhs file:  het_data, mask_data, j_max, seq_length, num_hets, num_masks, M_sequence_binned, M_vals, R_sequence_binned, R_vals
    mhs_files_M_file = {}
    mhs_files_M_file[i] = 'null'
    mhs_files_R_file = {}
    mhs_files_R_file[i] = 'null'
    sequences_info = bin_sequence(i,bin_size,mhs_files_M_file,mhs_files_R_file)
    hets = sequences_info[0][0,:]
    if sequences_info[1].shape[1]!=0:  # masked bases exist    
        het_sequence = np.zeros(int(sequences_info[3]/bin_size),dtype=int)
        mask_sequence = np.zeros(int(sequences_info[3]/bin_size),dtype=int)
        het_sequence[sequences_info[0][0]] = sequences_info[0][1]
        j_max = sequences_info[2]

        if bin_size <= j_max:
            print(f'ERROR; bin_size={bin_size}<j_max={j_max} which is not good.Aborting')
            sys.exit()
        else:
            mask_sequence[sequences_info[1][0]] = sequences_info[1][1]*bin_size
        sequence = het_sequence + mask_sequence
        E = write_emission_masked_probs(D,bin_size,theta,j_max,T,midpoint_end=midpoint_emissions) 
    else:
        
        sequence = np.zeros(shape=hets[-1]+1,dtype=int)
        sequence[hets] = 1
    

    # ii = '/home/tc557/rds/hpc-work/cobraa_snakemakes/simulate_from_HMM_231123/231125_panmixia_vs_structure/L100000000.0_modelpanmixia_true_D100_theta0.001_rho0.0008_gamma0.2_ts26_te65_spread10.1_spread250_sample10_chrom1.mhs'

    # mhs = pd.read_csv(i,delimiter='\t',header=None)
    # hets = np.array(mhs[1])

    B_sequence = np.zeros(shape=hets[-1]+1,dtype=int)
    R_sequence = np.zeros(shape=hets[-1]+1,dtype=int)
    B_vals = np.array([1])
    T_midpoints = np.array([(T[i]+T[i+1])/2 for i in range(0,len(T)-1)])
    forward, scales = forward_matmul_scaled_fcn(sequence=sequence,D=D,init_dist=init_dist,E=E,Q=Q_array,bin_size=bin_size,theta=theta,midpoints=T_midpoints,B_sequence=B_sequence,B_values=B_vals,R_sequence=R_sequence)
    LL = np.sum(np.log(scales))  
    if WETM != '':
        backward = backward_matmul_scaled_fcn(sequence=sequence,D=D,E=E,Q=Q_array,bin_size=bin_size,theta=theta,midpoints=T_midpoints,scales=scales,B_sequence=B_sequence,B_values=B_vals,R_sequence=R_sequence) 
        emissions_sequence = E[:,sequence[1:]]*B_vals[B_sequence[1:]]
        b_emissions = np.multiply(backward[:,1:],emissions_sequence)
        combined_forwardbackward = np.matmul(forward[:,0:-1],b_emissions.T)
        Q_current = Q_array[0]
        A_evidence = np.multiply(combined_forwardbackward,Q_current)
        np.savetxt(WETM,A_evidence)
        print(f'\tsaved transition evidence matrix to {WETM}')
else:
    coal_data = np.loadtxt(i)
    seqlen = coal_data[-1,1]
    markov_seq = np.zeros(int(seqlen),dtype=int)
    for j in range(0,coal_data.shape[0]):
        markov_seq[int(coal_data[j,0]):int(coal_data[j,1])] = int(coal_data[j,2])
    LL = markov_LL(markov_seq,Q)
print(f'LL (log-likelihood) is {LL}')

# python /home/tc557/cobraa/get_LL.py -i /home/tc557/rds/hpc-work/cobraa_snakemakes/simulate_from_HMM_231123/231125_panmixia_vs_structure/L100000000.0_modelpanmixia_true_D100_theta0.001_rho0.008_gamma0.5_ts26_te52_spread10.1_spread250_sample4_chrom1_tmrca.txt.gz -D 100 -spread_1 0.1 -spread_2 50 -bin_size 1 -rho 0.0008 -theta 0.001 -ts 26 -te 52 -gamma 0.3 
# python /home/tc557/cobraa/get_LL.py -i /home/tc557/rds/hpc-work/cobraa_snakemakes/simulate_from_HMM_231123/231125_panmixia_vs_structure/L100000000.0_modelpanmixia_true_D100_theta0.001_rho0.008_gamma0.5_ts26_te52_spread10.1_spread250_sample4_chrom1.mhs -D 100 -spread_1 0.1 -spread_2 50 -bin_size 1 -rho 0.0008 -theta 0.001 -ts 26 -te 52 -gamma 0.3 -HMM_LL