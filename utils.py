from transition_matrix import *
from BaumWelch import *
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
import pickle
from joblib import Parallel, delayed
from scipy import linalg


# Parse the given time segment pattern for lambda parameters
def write_segments(segment_pattern,D):
    # segment_pattern - either None or str. If str, each segment is delimited by comma and must sum to D
    # D - int. Number of discrete time windows
    if segment_pattern is None:
        segment_pattern = f"{D}*1,"
        segment_pattern = segment_pattern[0:-1]
    segments = [j.split('*') for j in segment_pattern.split(',')]
    segments_ints = [[int(i[0]),int(i[1])] for i in segments]
    sum_segments = 0
    for i in segments_ints:
        if i[1]!=0:
            sum_segments += i[0]*i[1]
        else:
            sum_segments += i[0]
    if sum_segments!=D:
        print(f'\tproblem in segment_pattern; sum of segment lengths={sum([i[0] for i in segments_ints])} which is not equal to D={D}.Aborting')
        sys.exit()
    return segments_ints


def get_het_mask_indices(in_path,bin_size):
    # in_path - str. String that points to the mhs file
    # bin_size - int. Number of basepairs to assume as block
    data = pd.read_csv(in_path, header = None,sep='\t') # load data
    hets = data[1].values -1 # read hets position, 0 indexed

    hets_diffs = [hets[i+1]-hets[i] for i in range(0,len(hets)-1)] # difference in bps between each het
    seq_length = hets[-1] + int(np.mean(hets_diffs)) # estimate seq_length; assume the last homozygous strech is of length (mean_difference_between_hets)
    # seq_length = hets[-1] + 2*bin_size
    hets_sequence = np.zeros(seq_length,dtype=int)
    masks_sequence = np.zeros(seq_length,dtype=int)
    hets_sequence[hets] = 1

    masks = [] # initialise empty list for mask indices
    ss_1 = data[1].loc[1:].values # het positions
    ss_2 = data[1].loc[0:data.shape[0]-2].values # het positions shifted
    ch = data[2].loc[1:].values # called homozygous 

    potential_masks = ss_1 - ch - ss_2 # mask reqiured, 0 if no, int>0 if yes

    # add masks
    if data[1].loc[0] - data[2].loc[0] > 0:
        start_mask = np.array([data[1].loc[0] - data[2].loc[0]])
        potential_masks = np.concatenate([start_mask,potential_masks])
        # ss_2 = np.concatenate([np.array([0]),ss_2])
        ss_2 = np.concatenate([np.array([0]),ss_2]) 

    mask_indices = np.where(potential_masks>0)[0]

    for i in mask_indices:
        masks_sequence[ss_2[i] : ss_2[i]+potential_masks[i]  ] = 1

    all_masks_positions = np.where(masks_sequence==1)[0]
    
    intersecting_indices = set(all_masks_positions).intersection(set(hets))
    if len(intersecting_indices) >0:
        print(f'Problem in fcn bin_sequence. Some indices are labelled as both hets and masks, these are \n{intersecting_indices}\nThis means there is a problem in your mhs file. Aborting')
        sys.exit()
    return hets_sequence, masks_sequence, seq_length, hets, all_masks_positions

def bin_sequence(in_path,step_size,mhs_files_B_file,mhs_files_R_file): 
    # in_path - str. String that points to the mhs file
    # step_size - int. Number of basepairs to assume as block
    # mhs_files_B_file - TODO
    # mhs_files_R_file - TODO
    printstring = f'\tfile={in_path}'
    # print(f'For file={in_path}',)
    hets_sequence, masks_sequence, seq_length, hets, masks = get_het_mask_indices(in_path,step_size) # hets_sequence is sequence of observation, masks_sequence is sequence of masks, seq_length is length of sequence, hets is indices of hets, masks is indices of masks
    
    # get B_val sequence
    B_file = mhs_files_B_file[in_path]
    # check B_file length is somewhat similar to mhs length
    B_sequence = get_B_sequence(B_file,seq_length,step_size)
    B_sequence_binned, B_vals = get_binned_data_B(B_sequence,step_size)
    printstring += f' sequence length is {seq_length}, number of hets is {len(hets)}'

    # get R_val sequence
    R_file = mhs_files_R_file[in_path]
    # check B_file length is somewhat similar to mhs length
    R_sequence = get_B_sequence(R_file,seq_length,step_size,ztype='R')
    roundme=-1 if np.min(R_sequence)==np.max(R_sequence) else 100
    R_sequence_binned, R_vals = get_binned_data_B(R_sequence,step_size,roundme=roundme)
    # printstring += f' sequence length is {seq_length}, number of hets is {len(hets)}'


    num_hets = len(hets)
    num_masks = len(masks)

    seq_length_binned = int(seq_length/step_size)
    het_data = get_binned_data(hets,step_size)
    mask_data = get_binned_data(masks,step_size)

    j_max = np.max(het_data[1,:]) # maximum number of mutations seen
    if j_max == step_size:
        print('Warning. Maximum number of mutations seen in bin is equal to bin_size. I think this should be ok.')
        sys.exit
    print(printstring,flush=True)
    # het data is array of two rows: first row is index, second row is corresponding number of hets in that index
    # mask data is array of two rows: first row is index, second row is corresponding number of masked bps in that index
    # j_max is maximum number of hets per bin
    # B_sequence_binned is an array of indices (ints), where the index corresponds with the b value in B_vals; ie B_sequence = [11,3,5] corresponds to the first three bins having a B value of [B_vals[11],B_vals[3],B_vals[5]] 
    return [het_data, mask_data, j_max, seq_length, num_hets, num_masks, B_sequence_binned, B_vals, R_sequence_binned, R_vals]



def get_binned_data(indices,bin_size):
    indices_binned = np.array(indices/bin_size,dtype=int) # get index of het if we bin into blocks of size bin_size
    indices_binned = np.squeeze(indices_binned) # get rid of outer dimension
    (zunique, zcounts) = np.unique(indices_binned, return_counts=True) # get number of hets per block
    binned_data = np.array([zunique,zcounts],dtype=int) # array describing pos and type of hets
    return binned_data

def decimate_values(zarray,roundme=1000):
    zarrayz_ = np.array(roundme*zarray,dtype=int)
    zarrayz = np.array(zarrayz_/roundme,dtype=float)
    if np.min(zarrayz)==0:
        zarrayz[zarrayz==np.min(zarrayz)] = 1e-5
    return zarrayz


def get_binned_data_B(B_sequence,bin_size,roundme=-1):
    seq_length = B_sequence.shape[0]
    seq_length_binned = int(seq_length/bin_size)
    B_sequence_ = B_sequence[0:B_sequence.shape[0] - B_sequence.shape[0]%bin_size]
    B_sequence_binned = np.average(B_sequence_.reshape(-1,bin_size),axis=1)
    if roundme!=-1:
        B_sequence_binned = decimate_values(B_sequence_binned,roundme=roundme)
    zB_sequence_binned = np.ones(B_sequence_binned.shape,dtype=int)
    B_vals = np.unique(B_sequence_binned)
    # if number_downsamples!=-1:
        # print(f'len(B_vals)={len(B_vals)}')
        # B_sequence_binned_rounded = decimate_values(B_sequence_binned,number_downsamples,roundme=100)
        # zB_sequence_binned = np.ones(B_sequence_binned_rounded.shape,dtype=int)
        # B_vals = np.unique(B_sequence_binned_rounded)
        # print(f'len(new_B_vals)={len(B_vals)}')
        
    for i in range(0,len(B_vals)):
        indices = np.where(B_sequence_binned==B_vals[i])[0]
        zB_sequence_binned[indices]=i    
    B_sequence_binned=zB_sequence_binned
    # zB_sequence_binned = [np.where(B_vals==i)[0][0] for i in B_sequence_binned]
    return B_sequence_binned, B_vals
    
    # get all indices in one bin, eg [1,2,3,4,5],[6,7,8,9,10]
    # take mean of B values of those indices
    # reshape and average

            # Y = np.arange(grid_start, grid_end + step_size, step_size)

            # # Map each element of X to the closest element in Y
            # rounded_X = Y[np.argmin(np.abs(X[:, np.newaxis] - Y), axis=1)]

# write coalescent time intervals according to the discretisation as introduced by Li & Durbin, Nature, 2011
def time_intervals(D,spread_1,spread_2,final_T_factor=None): 
    # D - int. Number of discrete time intervals
    # spread_1 - float. Describes spread of early time intervals
    # spread_2 - float. Describes spread of late time intervals
    
    T = [0]
    if final_T_factor is not None: # last boundary is a factor of second-to-last boundary
        for i in range(0,D-1): 
            T.append( spread_1*np.exp( (i/D)*np.log(1 + spread_2/spread_1) - 1))
        T.append(T[-1]*final_T_factor) # append  large last tMRCA to represent infinity
    else: # last boundary follows the sequence
        for i in range(0,D): 
            T.append( spread_1*np.exp( (i/D)*np.log(1 + spread_2/spread_1) - 1))
    T_np = np.array(T)
    return T_np

# def time_intervals(D,spread_1,spread_2): 
#     T = [0]
#     for i in range(0,D-1): 
#         T.append( spread_1*np.exp( (i/D)*np.log(1 + spread_2/spread_1) - 1))
#     T.append(T[-1]*4) # append stupidly large last tMRCA to represent infinity
#     T_np = np.array(T)
#     return T_np

# get log likelihood of sequence, with the current set of model parameters (theta, rho, lambda parameters)
def get_loglikelihood(BW,output_path):
    sequence, B_sequence, B_vals,R_sequence, R_vals = BW.sequence_fcn(0)
    tm_dummy = Transition_Matrix(D=BW.D,spread_1=BW.spread_1,spread_2=BW.spread_2,midpoint_transitions=BW.midpoint_transitions) # initialise transition matrix object
    tm_dummy.write_tm(lambda_A=BW.lambda_A_current,lambda_B=None,T_S_index=None,T_E_index=None,gamma=None,check=True,rho=BW.rho,exponential=not BW.recombnoexp) # write transition matrix for different rho values
    # Q_current_array = write_Q_array_withR(tm_dummy.Q,R_vals,R_vals[np.argmin(np.abs(R_vals-1))],BW.D)
    Q_current_array = write_Q_array_withR(tm_dummy.Q,R_vals,BW.rho,BW.D,BW.spread_1,BW.spread_2,BW.lambda_A_current,BW.midpoint_transitions) 


    
    # Q_current_array = np.zeros(shape=(len(R_vals),BW.D,BW.D))
    # tm_dummy = Transition_Matrix(D=BW.D,spread_1=BW.spread_1,spread_2=BW.spread_2,midpoint_transitions=BW.midpoint_transitions) # initialise transition matrix object
    # start = time.time()
    # for j in range(0,len(R_vals)):
    #     if j%5000==0:
    #         print(f'on {j} of {len(R_vals)}')
    #     Q_current_array[j,:,:] = tm_dummy.write_tm(lambda_A=BW.lambda_A_current,lambda_B=None,T_S_index=None,T_E_index=None,gamma=None,check=True,rho=BW.rho*R_vals[j]) # write transition matrix for different rho values
    #     # Q_current_array[j,:,:] = tm_dummy.write_tm(lambda_A=lambda_A_current,lambda_B=None,T_S_index=None,T_E_index=None,gamma=None,check=True,rho=rho) # write transition matrix for different rho values
    # # Q_current_array[0:-1,:,:] = Q_current_array[-1,:,:]
    # end = time.time()
    # time_taken = end - start
    # print(f'\t\t\ttime taken to write different tms: {time_taken}',flush=True)
    
    expectation_steps = Parallel(n_jobs=BW.num_files, backend='loky')(delayed(calculate_transition_evidence)(BW.sequence_fcn,file,BW.D,BW.init_dist,BW.E_masked,Q_current_array,BW.theta,BW.rho,BW.bin_size,BW.j_max,BW.midpoints,BW.spread_1,BW.spread_2,BW.midpoint_transitions) for file in range(BW.num_files)) 
    new_ll = 0
    for i in range(BW.num_files):
        new_ll += expectation_steps[i][1]

    final_lambda_A = BW.lambda_A_current
    if BW.T_S_index is not None:
        final_lambda_B = BW.lambda_B_current
        final_gamma = [BW.gamma_current]
    final_Q = BW.Q_current
    final_ll = new_ll
    final_LL_str = f'final log likelihood = {new_ll}'
    change_LL = new_ll - BW.LLs[-1]
    final_change_LL_str = f'final change in log likelihood = {change_LL}'
    iterations_str = f'number of iterations taken = {BW.number_of_completed_iterations}'
    theta_str = f'theta=4*N_E*mu = {BW.theta}'
    rho_str = f'rho=4*N_E*r = {BW.rho/BW.bin_size}'

    final_info_strings = [final_LL_str,final_change_LL_str,iterations_str,theta_str,rho_str]
    for s in final_info_strings:
        print(f'\t{s}')
        
    scaled_time = 0.5*BW.T_array*(BW.theta) # scale this to gens by dividing by mu. time_gens = scaled_time / mu
    scaled_inverse_lambda =  (4*BW.lambda_A_current)/BW.theta # scale this to inverse pop sizes with N_E = (1/scaled_inverse_lambda)/mu 
    
    ltb = scaled_time[0:BW.D]
    rtb = scaled_time[1:BW.D+1]

    
    scaletime_str = 'scale time by dividing by mu'
    scalelambda_str = 'scale lambda by taking its inverse then dividing by mu'
    final_info_strings.append(scaletime_str)
    final_info_strings.append(scalelambda_str)

    final_file = output_path + 'final_parameters.txt'
    
    if BW.T_S_index is None: # panmixia
        final_array = np.zeros(shape=(BW.D,3))
        final_array[:,0] = ltb
        final_array[:,1] = rtb 
        final_array[:,2] = scaled_inverse_lambda 
        columns_str = 'col 0 is left time boundary; col 1 is right time boundary; col 3 is scaled_lambda_A'
        footer = ''
    elif BW.T_S_index is not None: # structure 
        final_array = np.zeros(shape=(BW.D,4))
        final_array[:,0] = ltb
        final_array[:,1] = rtb 
        final_array[:,2] = scaled_inverse_lambda 
        scaled_inverse_lambda_B =  (4*BW.lambda_B_current)/BW.theta # scale this to inverse pop sizes with N_E = (1/scaled_inverse_lambda)/mu 
        final_array[:,3] = scaled_inverse_lambda_B 
        columns_str = 'col 0 is left time boundary; col 1 is right time boundary; col 3 is scaled_lambda_A; col 4 is scaled_lambda_B'
        footer = f'gamma is {BW.gamma_current}'
    final_info_strings.append(columns_str)
    header = "\n".join(final_info_strings)
    np.savetxt(final_file,final_array,comments='# ',header=header,footer=footer)
    print(f'\n\tFinal output saved to {final_file}')

    return None

def write_marginal_recomb_probs(BW,forward,backward,Q_current_array,sequence,B_sequence, B_vals,R_sequence, R_vals,output_R_path):
    emissions_sequence = BW.E_masked[:,sequence[1:]]
    b_emissions = np.multiply(backward[:,1:],emissions_sequence)
    zA = np.zeros(shape=(forward.shape[1]-1,BW.D,BW.D))
    binary_recomb = np.zeros(shape=(3,forward.shape[1]-1)) # first row position, second row prob of recomb, third row prob no recomb
    for i in range(0,forward.shape[1]-1):
        F = forward[:,i,np.newaxis]
        B = b_emissions[np.newaxis,:,i]
        zA[i,:,:] = np.matmul(F,B)*Q_current_array[R_sequence[i+1],:,:]
        binary_recomb[0,i] = i*BW.bin_size
        binary_recomb[1,i] = np.sum(zA[i,:,:]) - np.sum(np.diag(zA[i,:,:]))
        binary_recomb[2,i] = np.sum(np.diag(zA[i,:,:]))
    binary_recomb[1:,:] = binary_recomb[1:,:]/binary_recomb[1:,:].sum(axis=0)
    if np.max(R_vals)==np.min(R_vals):
        zstring='not '
    else:
        zstring=''
    zheader = f'Recombination map {zstring}given\nFirst row is position, second row is probability of recombining, third row is probability of not recombining\nTODO this assumes the recombination changes the state, which ignore the self coalescences (the SMCprime model)'
    np.savetxt(output_R_path, binary_recomb,comments='# ',header=zheader)
    print(f'\tsaved marginal recomb probability to {output_R_path}')
    return None


def get_posterior(BW,downsample,output_path,output_R_path):
    # this computes the posterior probability
    # the first row is the starting position of the current block of given bin size
    sequence, B_sequence, B_vals,R_sequence, R_vals = BW.sequence_fcn(0)

    tm_dummy = Transition_Matrix(D=BW.D,spread_1=BW.spread_1,spread_2=BW.spread_2,midpoint_transitions=BW.midpoint_transitions) # initialise transition matrix object
    tm_dummy.write_tm(lambda_A=BW.lambda_A_current,lambda_B=None,T_S_index=None,T_E_index=None,gamma=None,check=True,rho=BW.rho,exponential=not BW.recombnoexp) # write transition matrix for different rho values
    # Q_current_array = write_Q_array_withR(tm_dummy.Q,R_vals,R_vals[np.argmin(np.abs(R_vals-1))],BW.D)
    Q_current_array = write_Q_array_withR(tm_dummy.Q,R_vals,BW.rho,BW.D,BW.spread_1,BW.spread_2,BW.lambda_A_current,BW.midpoint_transitions) 


    # Q_current_array = np.zeros(shape=(len(R_vals),BW.D,BW.D))
    # tm_dummy = Transition_Matrix(D=BW.D,spread_1=BW.spread_1,spread_2=BW.spread_2,midpoint_transitions=BW.midpoint_transitions) # initialise transition matrix object
    # start = time.time()
    # for j in range(0,len(R_vals)):
    #     if j%5000==0:
    #         print(f'on {j} of {len(R_vals)}')
    #     Q_current_array[j,:,:] = tm_dummy.write_tm(lambda_A=BW.lambda_A_current,lambda_B=None,T_S_index=None,T_E_index=None,gamma=None,check=True,rho=BW.rho*R_vals[j]) # write transition matrix for different rho values
    # end = time.time()
    # time_taken = end - start
    # print(f'\t\t\ttime taken to write different tms: {time_taken}',flush=True)
    

    forward, scales = forward_matmul_scaled_fcn(sequence=sequence,D=BW.D,init_dist=BW.init_dist,E=BW.E_masked,Q=Q_current_array,bin_size=BW.bin_size,theta=BW.theta,midpoints=BW.midpoints,B_sequence=B_sequence,B_values=B_vals,R_sequence=R_sequence)
    backward = backward_matmul_scaled_fcn(sequence=sequence,D=BW.D,E=BW.E_masked,Q=Q_current_array,bin_size=BW.bin_size,theta=BW.theta,midpoints=BW.midpoints,scales=scales,B_sequence=B_sequence,B_values=B_vals,R_sequence=R_sequence) 
    posterior = np.multiply(forward,backward)
    posterior = posterior/posterior.sum(axis=0) # normalise

    if output_R_path is not None:
        write_marginal_recomb_probs(BW,forward,backward,Q_current_array,sequence,B_sequence, B_vals,R_sequence, R_vals,output_R_path)

    length = sequence.shape[0]
    ll = np.sum(np.log(scales)) 
    print(f'\tlog likelihood is {ll}')
    position_array = np.arange(0,length,1)*BW.bin_size
    position_posterior = np.zeros((posterior.shape[0]+1,posterior.shape[1])) 
    position_posterior[0,:] = position_array
    position_posterior[1:,:] = posterior
    theta_str = f'theta=4*N_E*mu = {BW.theta}'
    rho_str = f'rho=4*N_E*r = {BW.rho/BW.bin_size}'
    binsize_str = f'bin_size = {BW.bin_size}'
    description_str = f'first row is position'
    LL_str = f'log likelihood is {ll}'
    time_array = BW.T_array
    time_array_string = ",".join([str(i) for i in time_array])
    final_info_strings = [theta_str,rho_str,binsize_str,description_str,LL_str,time_array_string]
    header = "\n".join(final_info_strings)
    if downsample>1:
        length = int(position_posterior.shape[1]/downsample)
        position_posterior_downsample = np.zeros((position_posterior.shape[0],length))
        for i in range(0,length):
            position_posterior_downsample[:,i] = position_posterior[:,i*downsample]
        position_posterior = position_posterior_downsample
    np.savetxt(output_path, position_posterior,comments='# ',header=header)
    print(f'\tsaved posteriors to {output_path}')

    return None

def get_B_sequence(B_file,seq_length,bin_size,ztype='B'):
    if B_file=='null':
        B_sequence = np.ones(seq_length,dtype=int)
    else:
        B_sequence = np.zeros(seq_length,dtype=float) - 1
        # B_sequence = np.zeros(seq_length,dtype=float)

        # my files, eg in: /n/scratch3/users/t/trc468/B_stat/my_bkgd
        # B_stat = np.loadtxt(B_file)

        # Arun's files, eg in: /n/scratch3/users/a/ard063/bkgd_bed/
        B_data = pd.read_csv(B_file, header = None,sep='\t') # load data TODO
        B_stat = np.array(B_data.loc[:,1:3])
        b_length = B_stat[:,1][-1]
        length_diff_thresh=5e+06
        endcheck=False
        if seq_length>b_length+length_diff_thresh:
            print(f'\tLength of mhs file = {seq_length} is more than length than {ztype}_map file ={b_length} + threshold = {length_diff_thresh} ; {ztype}_file = {B_file}. Aborting.',flush=True)
            sys.exit()
        elif seq_length>b_length and seq_length<b_length+length_diff_thresh:
            print(f'\tLength of mhs file = {seq_length}; length of {ztype}_map file ={b_length}; padding {seq_length-b_length} base pairs with {ztype}=1',flush=True)
            padcheck=seq_length-b_length
            endcheck=True
        else:
            print(f'\tLength of mhs file = {seq_length}; length of {ztype}_map file ={b_length}; trimming {b_length-seq_length} base pairs',flush=True)

        z = np.copy(B_stat[:,2])
        z2 = np.copy(B_stat[:,0:2])
        B_stat[:,0]=z
        B_stat[:,1:]=z2

        # B_vals = np.unique(B_stat[:,0])
        # B = [np.where(B_vals==i)[0][0] for i in B_stat[:,0]]
        # B_stat[:,0] = B
        prev=0
        for k in range(0,B_stat.shape[0]):
        # k=0
        # while B_sequence[-1]!=-1:
            if B_stat[k,1]!=prev:
                print(f'\tProblem. There is an unannotated gap in {ztype}__file={B_file} at line={k}; nearest_index={prev}.Aborting',flush=True)
                sys.exit()
            # zB_sequence[int(B_stat[k,1]):int(B_stat[k,2])] = B_stat[k,0]

            B_sequence[int(B_stat[k,1]):int(B_stat[k,2])] = B_stat[k,0]
            prev=B_stat[k,2]
            if int(B_stat[k,2])>seq_length:
                continue # exclude trailing base pairs
        if endcheck==True:
            padme = np.where(B_sequence<0)[0]
            if padcheck!=len(padme):
                print('Problem with end of {ztype} file. Aborting.')
                sys.exit()
            B_sequence[padme]=1
        if len(np.where(B_sequence<0)[0])>0:
            print(f'\tProblem. There are negative {ztype} values in {ztype}_sequence. This could be a problem in differing length between mhs and {ztype} file.Aborting')
            sys.exit()
    return B_sequence




# write emission probabilities
def write_emission_probs(D,L,theta,j_max,T,m=0,midpoint_end=True):
    
    # write emissions with Poisson; rate theta*L*t
    # D is number of hidden states
    # L is bin_size
    # mu is per gen per bp mutation rate
    # theta is scaled recomb rate, theta = 2*N*mu where 2N is the effective diploid size and mu is the per gen per bp mutation rate
    # j_max is maximum number of mutations seen
    # T is array of D times in coalescent time
    # take midpoints of each interval
    # m is number of masks in bin
    if m>L:
        print(f'Problem in fcn write_emission_probs. m={m}>L={L}')
    midpoints = np.array([(T[i]+T[i+1])/2 for i in range(0,len(T)-1)])
    
    if midpoint_end is False:
        midpoints[-1] = (T[-2]+T[-2]+3)/2
    E = np.zeros(shape=(D,j_max+1))
    for j in range(0,j_max+1):
        # print('shape of abba is {}'.format(abba.shape))
        # print('shape of E[:,j] is {}'.format(E[:,j].shape))
        # E[:,j] = np.array([((( L*theta*midpoints[i])**j)*np.exp(-L*theta*midpoints[i]))/np.math.factorial(j) for i in range(0,D)]) # old, no masks
        E[:,j] = np.array([((( (L-m)*theta*midpoints[i])**j)*np.exp(-(L-m)*theta*midpoints[i]))/np.math.factorial(j) for i in range(0,D)]) # update220117_1825
    
    return E

def write_emission_masked_probs(D,L,theta,j_max,T,midpoint_end):
    # write as 2 dimensional object: E_masked[(num_hets,num_masks),num_states] - 
    # e.g. if bin_size = 100 and bin b has 35 masks and 4 hets, then the index will be 35*bin_size + 4 = 3504
    # e.g. if bin_size = 100 and bin b has 1 mask and 0 hets, then the index will be 1*bin_size + 0 = 100
    # e.g. if bin_size = 100 and bin b has 0 masks and 12 hets, then the index will be 0*bin_size + 12 = 12
    # e.g. if bin_size = 100 and bin b has 99 masks and 1 het, then the index will be 99*bin_size + 1 = 9901
    # thus of shape 
    # note num_masks + num_hets per bin cannot exceed bin_size
    if L<=j_max:
        E_masked = np.zeros(shape=( D,(L+1)*L + j_max+1))
        print(f'bin_size={L} < j_max ={j_max}')
        # mask_sequence[self.sequences_info[file][1][0]] = self.sequences_info[file][1][1]*(j_max+1)
        M = j_max+1
        for k in range(0,L+1): # different possible bin sizes
            # E_masked[(k*L):(k*L)+j_max,:] = write_emission_probs(D,L,theta,j_max,T,k)
            E_masked[:,(k*M):(k*M)+j_max+1] = write_emission_probs(D,L,theta,j_max,T,k,midpoint_end)
    else:
        
        E_masked = np.zeros(shape=( D,(L)*L + j_max+1)) # num masks, num states, num hets
        for k in range(0,L+1): # different possible bin sizes
            # E_masked[(k*L):(k*L)+j_max,:] = write_emission_probs(D,L,theta,j_max,T,k)
            E_masked[:,(k*L):(k*L)+j_max+1] = write_emission_probs(D,L,theta,j_max,T,k,midpoint_end)

    return E_masked

def write_emission_masked_Bvals_probs(D,L,theta,j_max,T,B_vals,midpoint_end):
    len_B = len(B_vals)
    if L<=j_max:
        E_masked_B = np.zeros(shape=(len_B,D,(L+1)*L + j_max+1))
        # E_masked = np.zeros(shape=(D,(L+1)*L + j_max+1))
        print(f'bin_size={L} < j_max ={j_max}')
        # mask_sequence[self.sequences_info[file][1][0]] = self.sequences_info[file][1][1]*(j_max+1)
        M = j_max+1
        for q in range(0,len(B_vals)): # different indices of B values
            for k in range(0,L+1): # different possible bin sizes
                # E_masked[(k*L):(k*L)+j_max,:] = write_emission_probs(D,L,theta,j_max,T,k)
                # E_masked[:,(k*M):(k*M)+j_max+1] = write_emission_probs(D,L,theta,j_max,T,k,midpoint_end)
                E_masked_B[q,:,(k*M):(k*M)+j_max+1] = write_emission_probs_b(D,L,theta,j_max,T,B_vals[q],k,midpoint_end)
    else:
        # E_masked = np.zeros(shape=( D,(L)*L + j_max+1)) # num masks, num states, num hets
        E_masked_B = np.zeros(shape=(len_B,D,(L)*L + j_max+1))
        for q in range(0,len(B_vals)):
            for k in range(0,L+1): # different possible bin sizes
                # E_masked[(k*L):(k*L)+j_max,:] = write_emission_probs(D,L,theta,j_max,T,k)
                # E_masked[:,(k*L):(k*L)+j_max+1] = write_emission_probs(D,L,theta,j_max,T,k,midpoint_end)
                E_masked_B[q,:,(k*L):(k*L)+j_max+1] = write_emission_probs_b(D,L,theta,j_max,T,B_vals[q],k,midpoint_end)
    return E_masked

def write_emission_probs_b(D,L,theta,j_max,T,b,m=0,midpoint_end=True):
    # write emissions with Poisson; rate theta*L*t
    # D is number of hidden states
    # L is bin_size
    # mu is per gen per bp mutation rate
    # theta is scaled recomb rate, theta = 2*N*mu where 2N is the effective diploid size and mu is the per gen per bp mutation rate
    # j_max is maximum number of mutations seen
    # T is array of D times in coalescent time
    # take midpoints of each interval
    # m is number of masks in bin
    # b is float for B value 

    if m>L:
        print(f'Problem in fcn write_emission_probs. m={m}>L={L}')
    midpoints = np.array([(T[i]+T[i+1])/2 for i in range(0,len(T)-1)])/b
    if midpoint_end is False:
        midpoints[-1] = (T[-2]+T[-2]+3)/2
    E = np.zeros(shape=(D,j_max+1))
    for j in range(0,j_max+1):
        # print('shape of abba is {}'.format(abba.shape))
        # print('shape of E[:,j] is {}'.format(E[:,j].shape))
        # E[:,j] = np.array([((( L*theta*midpoints[i])**j)*np.exp(-L*theta*midpoints[i]))/np.math.factorial(j) for i in range(0,D)]) # old, no masks
        E[:,j] = np.array([((( (L-m)*theta*midpoints[i])**j)*np.exp(-(L-m)*theta*midpoints[i]))/np.math.factorial(j) for i in range(0,D)]) # update220117_1825
    return E






def write_emission_masked_probs_old(D,L,theta,j_max,T,midpoint_end=True):
    E_masked = np.zeros(shape=(L+1,D,j_max+1)) # num masks, num states, num hets
    for k in range(0,L+1):
        E_masked[k,:,:] = write_emission_probs(D,L,theta,j_max,T,k)
    return E_masked


def get_num_masks_hets(x,bin_size):
    num_masks = abs(int(x/bin_size))
    num_hets = abs(x) - num_masks*bin_size
    return num_masks, num_hets

@njit
def multiply_through(D,forward,b_emissions,Q_current_array,R_sequence,A_evidence_new):
    for ii in range(0,D):
        for jj in range(0,D):
            A_evidence_new[ii,jj] = np.sum(forward[ii,0:-1]*b_emissions.T[:,jj]*Q_current_array[R_sequence[0:-1],ii,jj])
    return A_evidence_new

def calculate_transition_evidence(sequence_fcn,file,D,init_dist,E_masked,Q_current_array,theta,rho,bin_size,j_max,midpoints,spread_1,spread_2,midpoint_transitions,jump_size=int(1e+05),stride_width=1000):
# file is the label of file (int)


    # deletemedict = {}
    # deletemedict[0] = Q_current_array
    # deletemedictpath = '/home/trc468/deletemedict.pickle'
    # with open(deletemedictpath,'wb') as f : pickle.dump(Q_current_array,f)
    # with open(deletemedictpath,'rb') as f: deletemedict = pickle.load(f)
    # Q_current_array = deletemedict
    # Q_current_array[0:-1,:,:] = Q_current_array[-1,:,:]

    # if B_vals[0]==0: # TODO REMOVE THIS; problem with length of B_stat file and mhs file
        # B_vals[0]=1
    sequence, B_sequence, B_vals, R_sequence, R_vals = sequence_fcn(file)
    # calculate forward
    # forward, scales = forward_matmul_scaled_fcn(sequence=sequence,D=D,init_dist=init_dist,E=E_masked,Q=Q_current)        
    # ll = np.sum(np.log(scales))     
    # backward = backward_matmul_scaled_fcn(sequence=sequence,D=D,E=E_masked,Q=Q_current,scales=scales) # matmult, classic forward with matrix multiplication
    forward, scales = forward_matmul_scaled_fcn(sequence=sequence,D=D,init_dist=init_dist,E=E_masked,Q=Q_current_array,bin_size=bin_size,theta=theta,midpoints=midpoints,B_sequence=B_sequence,B_values=B_vals,R_sequence=R_sequence)
    backward = backward_matmul_scaled_fcn(sequence=sequence,D=D,E=E_masked,Q=Q_current_array,bin_size=bin_size,theta=theta,midpoints=midpoints,scales=scales,B_sequence=B_sequence,B_values=B_vals,R_sequence=R_sequence) # matmult, classic forward with matrix multiplication        
    ll = np.sum(np.log(scales))  
    # backward_numba_normalscales = backward_matmul_scaled_fcn_numba(sequence=sequence,D=D,E=E_masked,Q=Q_current,scales=scales) # matmult, classic forward with matrix multiplication

    # calculate A_evidence
    # A = np.ones(shape=E_masked[:,sequence[1:]].shape)*-1
    if max(B_vals)==min(B_vals):
        emissions_sequence = E_masked[:,sequence[1:]]*B_vals[B_sequence[1:]]
    else:
        emissions_sequence = np.array([write_emission_probs_b_slice(D,bin_size,theta,midpoints,B_vals[B_sequence[i]],sequence[i]) for i in range(1,len(B_sequence))]).T

    # emissions_sequence = E_masked[:,sequence[1:]]
    b_emissions = np.multiply(backward[:,1:],emissions_sequence)
    combined_forwardbackward = np.matmul(forward[:,0:-1],b_emissions.T)
    if max(R_vals)==min(R_vals):
        Q_current = Q_current_array[np.argmin(np.abs(R_vals-1)),:,:]
        A_evidence = np.multiply(combined_forwardbackward,Q_current)
    else:
        # A_evidence_new = np.zeros(shape=(D,D))
        # for ii in range(0,D):
        #     for jj in range(0,D):
        #         A_evidence_new[ii,jj] = np.sum(forward[ii,0:-1]*b_emissions.T[:,jj]*Q_current_array[R_sequence[0:-1],ii,jj])
        A_evidence = np.einsum('il, lj, lij->ij', forward[:,0:-1], b_emissions.T, Q_current_array[R_sequence[0:-1],:,:])

        # zA_evidence = np.zeros(shape=(len(R_vals),D,D))
        # for w in range(0,len(R_vals)):
        #     zindices = np.where(R_sequence==w)[0]
        #     zindices = zindices[zindices!=b_emissions.shape[1]]
        #     zcombined_forwardbackward = np.matmul(forward[:,zindices],b_emissions[:,zindices].T)
        #     zA_evidence[w,:,:] = np.multiply(zcombined_forwardbackward,Q_current_array[w,:,:])
        
    # return zA_evidence, ll
    # return A_evidence_newnew, ll
    # return A_evidence_new, ll
    return A_evidence, ll

@njit 
def forward_matmul_scaled_fcn(sequence,D,init_dist,E,Q,bin_size,theta,midpoints,B_sequence,B_values,R_sequence): # matmult, classic forward with matrix multiplication
    # L = len(sequence) # length of sequence
    # f_ = np.zeros(shape=(D,L)) # initialise forward variables
    # scales = np.zeros(L) # intialise array to store scaled values
    # scales[0]= sum(E[:,sequence[0]] * init_dist) # s(1)
    # f_[:,0] = (E[:,sequence[0]] * init_dist)/scales[0] # f_(1); Assume uniform initial distribution, s1 = 0
    # for i in range(0,L-1):
    #     emissions_transitions_pos = E[:,sequence[i+1]]*np.dot(f_[:,i],Q)
    #     s = sum(emissions_transitions_pos)
    #     scales[i+1] = s # add s to array of scales
    #     f_[:,i+1] = emissions_transitions_pos/s
    # return f_, scales
    L = len(sequence) # length of sequence
    f_ = np.zeros(shape=(L,D)) # initialise forward variables
    scales = np.zeros(L) # intialise array to store scaled values
    scales[0]= sum(E[:,sequence[0]] * init_dist) # s(1) 
    f_[0,:] = (E[:,sequence[0]] * init_dist)/scales[0] 
    # zQ = np.copy(Q[0,:,:])
    # scales[0]= sum(1 * init_dist) # s(1) # assume first element masked
    # f_[0,:] = (1 * init_dist)/scales[0] # assume first element masked
    for i in range(0,L-1):
        # if B_values[B_sequence[i+1]]<0.95:
        # emissions_transitions_pos = E[:,sequence[i+1]]*np.dot(f_[i,:],Q) 
        
        emissions_transitions_pos = write_emission_probs_b_slice(D,bin_size,theta,midpoints,B_values[B_sequence[i+1]],sequence[i+1])*np.dot(f_[i,:],Q[R_sequence[i+1],:,:].copy()) 
        # emissions_transitions_pos = write_emission_probs_b_slice(D,bin_size,theta,midpoints,B_values[B_sequence[i+1]],sequence[i+1])*np.dot(f_[i,:],zQ) 

        # zab = write_emission_probs_b_slice(D,bin_size,theta,midpoints,B_values[B_sequence[i+1]],sequence[i+1])
        # zba = np.dot(f_[i,:],Q[R_sequence[i+1],:,:].copy()) 
        # emissions_transitions_pos = zab*zba
        s = sum(emissions_transitions_pos)
        scales[i+1] = s # add s to array of scales
        f_[i+1,:] = emissions_transitions_pos/s
    return f_.transpose(), scales




@njit
def backward_matmul_scaled_fcn(sequence,D,Q,bin_size,theta,midpoints,E,scales,B_sequence,B_values,R_sequence):
    L = len(sequence) # length of sequence
    b_ = np.zeros(shape=(L,D)) # initialise backward variables, only storing present and previous
    b_[L-1,:] = np.ones(D)/scales[-1] # uniform 1 to initiatilse backward   
    for i in range(L-1):
        j = L-2-i
        # emissions_A = np.multiply(Q,E[:,sequence[j+1]]) # emissions*Q
        emissions_A = np.multiply(Q[R_sequence[j+1]],write_emission_probs_b_slice(D,bin_size,theta,midpoints,B_values[B_sequence[j+1]],sequence[j+1])) # emissions*Q
        b_[j,:] = np.dot(emissions_A,b_[j+1,:])/scales[j] # emissions*Q*b
    return b_.transpose()


@njit 
def write_emission_probs_b_slice(D,L,theta,midpoints,b,seq):
    # write emissions with Poisson; rate theta*L*t
    # D is number of hidden states
    # L is bin_size
    # mu is per gen per bp mutation rate
    # theta is scaled recomb rate, theta = 2*N*mu where 2N is the effective diploid size and mu is the per gen per bp mutation rate
    j = int(np.mod(seq,L)) # number of hets
    m = int(np.floor(seq/L)) # number of masks
    # midpoints=midpoints/b # old, tested this
    midpoints=midpoints*b # different to /home/trc468/SPSMC_bs_mistake_230115
    E = np.array([((( (L-m)*theta*midpoints[i])**j)*np.exp(-(L-m)*theta*midpoints[i]))/factorial_fcn(j) for i in range(0,D)])
    return E

# @njit
# def write_Q_array_withR_old(Qbase,R_vals,r,D):
#     zQ = np.zeros(shape=(len(R_vals),D,D))
#     for i,j in enumerate(R_vals):
#         zzQ = np.copy(Qbase)
#         np.fill_diagonal(zzQ,0) 
# #         zQ[range(D),range(D)] = 0 
#         zzQ = zzQ*R_vals[i]/r
#         zzQ[range(D),range(D)] = 1-zzQ.sum(axis=1)
#         zQ[i,:,:] = zzQ
#     return zQ

def write_Q_array_withR(Qbase,R_vals,rho,D,spread_1,spread_2,lambda_A,midpoint_transitions):
    T = time_intervals(D,spread_1,spread_2)
    e_betas=np.array([e_beta(D,T,lambda_A,None,None,None,None,int,midpoint_transitions) for int in range(0,D)]) # get expected time in each interval
    e_betas = e_betas[:,np.newaxis]
    Qbase_nodiag = np.copy(Qbase)
    np.fill_diagonal(Qbase_nodiag,0)
    recomb_probabilities = 1-np.exp(-rho*e_betas)
    Qbase_nodiag = Qbase_nodiag/recomb_probabilities
    zQ = zwrite_Q_array_withR(Qbase_nodiag,T,e_betas,R_vals,rho,D,spread_1,spread_2,lambda_A,midpoint_transitions)
    return zQ

@njit
def zwrite_Q_array_withR(Qbase_nodiag,T,e_betas,R_vals,rho,D,spread_1,spread_2,lambda_A,midpoint_transitions):
    zQ = np.zeros(shape=(len(R_vals),D,D))
    for i,j in enumerate(R_vals):
        zrecomb_probabilities = 1-np.exp(-R_vals[i]*rho*e_betas)
        zzQ = Qbase_nodiag*zrecomb_probabilities
        for qq in range(0,D):
            zzQ[qq,qq] = 1-zzQ[qq,:].sum()
        zQ[i,:,:] = zzQ
    return zQ


# def write_Q_array_withR(lambda_A,rho,R_vals,tm_dummy,D):
#     zQ = np.zeros(shape=(len(R_vals),D,D))
#     for j in range(0,len(R_vals)):
#         zQ[j,:,:] = tm_dummy.write_tm(lambda_A=lambda_A,lambda_B=None,T_S_index=None,T_E_index=None,gamma=None,check=True,rho=rho*R_vals[j]) # write transition matrix for different rho values
#     return zQ


@njit
def factorial_fcn(x):
    prod=1
    for i in range(1,x+1):
        prod*=i
    return prod

def downsample_r(sequences_info,num_files_R,number_downsamples_R,num_files):
    if num_files_R==0:
        return sequences_info
    else:       
        all_r = []
        for j in range(0,num_files_R):
            all_r += sequences_info[j][9].tolist()
        all_r = np.sort(np.array(all_r))
        if len(all_r)==1:
            return sequences_info
        else:    
            window_size = len(all_r)//number_downsamples_R
            number_downsamples_R = len(all_r)//window_size + 1

            mean_R_array = np.zeros(number_downsamples_R)
            for i in range(0,number_downsamples_R):
                mean_R_array[i] = np.mean(all_r[window_size*i:window_size*(i+1)])
            for j in range(0,num_files):
                A = np.zeros(len(sequences_info[j][9]),dtype=int)
                for q in range(0,len(sequences_info[j][9])):
                    A[q] = np.argmin(np.abs(mean_R_array-sequences_info[j][9][q]))
                sequences_info[j][8] = A[sequences_info[j][8]]
                sequences_info[j][9] = mean_R_array
            return sequences_info


def multiply_me(zQ,freqs):
    zzQ = np.copy(zQ)
    for i,j in enumerate(freqs):
        zzQ[i,:,:] = zQ[i,:,:]*j
    return zzQ