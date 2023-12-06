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


# parse args

parser = argparse.ArgumentParser(description="Inputs and parameters")
parser.add_argument('-in','--input_file_path',help='The path(s) to mhs file(s)',type=str,nargs='+')
parser.add_argument('-in_M','--input_file_path_M',help='The path(s) to associated mutation rate file(s); if given must be given in same order as mhs files',type=str,nargs='+')
parser.add_argument('-in_R','--input_file_path_R',help='The path(s) to associated recomb rate file(s); if given must be given in same order as mhs files',type=str,nargs='+')
parser.add_argument('-o','--file_outpath',help='Output prefix',default=None,required=False)
parser.add_argument('-D','--number_time_windows',help='The number of time windows to use in inference',nargs='?',const=32,type=int,default=30)
parser.add_argument('-spread_1','--D_spread_1',help='Parameter controlling the time interval boundaries',nargs='?',const=0.1,type=float,default=0.1)
parser.add_argument('-spread_2','--D_spread_2',help='Parameter controlling the time interval boundaries',nargs='?',const=50,type=float,default=50)
parser.add_argument('-b','--bin_size',help='Adjust recombination rate to bin this many bases together', nargs='?',const=100,type=int,default=100)
parser.add_argument('-rho','--scaled_recombination_rate',help='The scaled recombination rate; if p is per gen per bp recombination rate, and 2N is the diploid effective size, rho =4Np',nargs='?',const=0.0004,type=float,default=0.0004)
parser.add_argument('-theta','--scaled_mutation_rate',help='The scaled mutation rate; if mu is per gen per bp mutation rate, and 2N is the diploid effective size, theta =4Nmu',required=False,type=str,default=None)
parser.add_argument('-rho_fixed','--rho_fixed',help='Boolean for optimising rho as a parameter',default=False,action='store_true')
parser.add_argument('-mu_over_rho_ratio','--mu_over_rho_ratio',help='Starting ratio between theta and rho',nargs='?',const=False,type=float,default=None)
parser.add_argument('-lambda_lwr','--lambda_lwr_bnd',help='Lower bound for lambda when searching for psc parameters',nargs='?',const=False,type=float,default=0.1)
parser.add_argument('-lambda_upr','--lambda_upr_bnd',help='Upper bound for lambda when searching for psc parameters',nargs='?',const=False,type=float,default=50)

parser.add_argument('-lambda_lwr_struct','--lambda_lwr_bnd_struct',help='Lower bound for lambda when searching for psc parameters in structured period',nargs='?',const=False,type=float,default=0.4)
parser.add_argument('-lambda_upr_struct','--lambda_upr_bnd_struct',help='Upper bound for lambda when searching for psc parameters in structured period',nargs='?',const=False,type=float,default=2)

parser.add_argument('-lambda_A_fg','--lambda_A_fg',help='First guess for lambda_A',nargs='?',const=False,type=str)
parser.add_argument('-lambda_B_fg','--lambda_B_fg',help='First guess for lambda_A',nargs='?',const=False,type=str)
parser.add_argument('-lambda_A_segments','--lambda_A_segments',help='time segment pattern for lambda A',nargs='?',const=False,type=str)
parser.add_argument('-lambda_B_segments','--lambda_B_segments',help='time segment pattern for lambda B',nargs='?',const=False,type=str)
parser.add_argument('-xtol','--xtol',help='tolerance in solution to optimisation for Powell',type=float, required=False,default=0.0001)
parser.add_argument('-ftol','--ftol',help='tolerance in optimisation function for Powell',type=float, required=False,default=0.0001)

parser.add_argument('-number_downsamples_R','--number_downsamples_R',help='Number of points in R to take',nargs='?',const=False,type=int,default=200)
parser.add_argument('-its','--BW_its',help='Number of iterations to be used in BaumWelch algorithm',nargs='?',const=20,type=int,default=20)
parser.add_argument('-thresh','--BW_threshold',help='Iterate the BaumWelch algorithm until the change in log-likelihood is lower than this',nargs='?',const=0.1,type=float,default=1)
parser.add_argument('-recombnoexp','--recombnoexp',help='Model for recombination probability; either exponential (approximation with Taylor series) or standard',default=False,action='store_true')

parser.add_argument('-midpoint_transitions','--midpoint_transitions',help='Whether to take midpoint in transitions',type=str, required=False,default="False")
parser.add_argument('-midpoint_emissions','--midpoint_emissions',help='Whether to take midpoint for the final two boundaries in the emission probabilities (take the midpoint at all other boundaries by default)',type=str, required=False,default="False")
parser.add_argument('-final_T_factor','--final_T_factor',help='If given, for the final time boundary take T[-2]*factor. Otherwise write according to sequence',type=str, required=False,default="False")

parser.add_argument('-optimisation_method','--optimisation_method',help='Whether optimisation method',type=str, required=False,default="Powell")
parser.add_argument('-save_iteration_files','--save_iteration_files',help='Flag for whether to save output for every iteration',default=False,action='store_true')
parser.add_argument('-decode','--decode_flag',help='Flag for whether to run posterior decoding (as opposed to demographic inference)',default=False,action='store_true')
parser.add_argument('-decode_downsample','--decode_downsample',help='If decode is true, downsammple the posterior positions by this factor',type=int,default=10,required=False)
parser.add_argument('-o_R','--output_R_path',help='If given with "decode" flag, save marginal recombination probabilities to this path',default=None,type=str,required=False)
parser.add_argument('-c','--cores',help='Number of cores; if not given, will try the number of mhsfiles',default=None,type=int,required=False)


"""
I explain the -lambda_fg and -lambda_segments parameters. The array of lambda values will always be of length D, it is written in the BaumWelch script.
However, sometimes one might want to group neighbouring parameters to be the same value. For example, suppose D=32, instead of having 32 free parameters, 
one might wish for the first 4 to be equal to each other, the last 2 to be equal to each other, and the middle 26 parameters to be free. One can select this with the 
-lambda_segments parameter by doing: -lambda_segments 1*4,26*1,1*2
which can be read as "1 lot of 4, 26 lots of 1, 1 lot of 2". 
The sum of these multiplied pairs must equal D, e.g. with the example above (1*4 + 26*1 + 1*2)=32

The default starting value for the lambda array is 1 everywhere. One can choose these values with -lambda_fg. If specified, the number of values given must be equal to 
the number of segments. Eg with D=12 -lambda_fg 1,1,1,1,1,1,1,1,2 -lambda_segments 8*1,1*4 gives lambda=[1,1,1,1,1,1,1,1,2,2,2,2]

Suppose you don't want to search for a particular parameter, then you can mark it as fixed in the -lambda_segments array with a "0". 
For example suppose D=12 and you want to search for the first 8 parameters, but leave the last 4 fixed at their starting value. Then you use
-lambda_segments 8*1,4*0
means the lambda_array will be [optimised,optimised,optimised,optimised,optimised,optimised,optimised,optimised,fixed,fixed,fixed,fixed]

You can also sepcify a different starting guess, then fix that too:
-lambda_fg = 1,1,1,1,2,3,3,3,3 -lambda_segments 4*1,4*0,4*1
gives lambda_array=[1,1,1,1,2,2,2,2,3,3,3,3], where the values at 2 are fixed.
"""
parser.add_argument('-gamma_lwr','--gamma_lwr_bnd',help='Lower bound for gamma when searching for struct parameters',nargs='?',const=False,type=float,default=0)
parser.add_argument('-gamma_upr','--gamma_upr_bnd',help='Upper bound for gamma when searching for struct parameters',nargs='?',const=False,type=float,default=1)
parser.add_argument('-gamma_fg','--gamma_fg',help='First guess for gamma.',nargs='?',const=False,type=float,default=0.2)
parser.add_argument('-ts','--T_S_index',help='Index of T_S, must be less than D (can be "None" for panmixia)',nargs='?', const='None',type=str,default='None')
parser.add_argument('-te','--T_E_index',help='Index of T_E, must be less than D (can be "None" for panmixia)',nargs='?', const='None',type=str,default='None')

args = parser.parse_args()



print(f'\nRunning cobraa; last updated 231031ymd; v1.1',flush=True)
arguments = sys.argv[1:]
command_line = 'python ' + ' '.join(['"{}"'.format(arg) if ' ' in arg else arg for arg in [sys.argv[0]] + arguments])
print(f'Command line: {command_line}')

if args.decode_flag is True:
    decode_flag = True
    inference_flag = False
    downsample = args.decode_downsample
    print(f'\tDecoding',flush=True)
    if downsample>1: print(f'\tdownsampling x{downsample}')
    if len(args.input_file_path)>1:
        print(f'decode can only take one file. Aborting. ',flush=True)
        sys.exit()
else:
    inference_flag = True

# input files (mhs)
num_files = len(args.input_file_path)
# mhs_files = np.zeros(shape=(num_files,))
files_paths = [file for file in args.input_file_path]

# input files (mutation rate)
if args.input_file_path_M==None:
    num_files_M = 0
else:
    num_files_M = len(args.input_file_path_M)

# input files (recomb rate)
if args.input_file_path_R==None:
    num_files_R = 0
else:
    num_files_R = len(args.input_file_path_R)

recombnoexp = args.recombnoexp
mhs_files_M_file = {}
if num_files_M>0:
    if num_files!=num_files_M:
        print(f'\tProblem. Mutation rate files provided, but number of files={num_files_M} does not match number of mhs files={num_files}. Aborting',flush=True)
        sys.exit()
    files_paths_M = [file for file in args.input_file_path_M]
    print(f'\nLoaded mhs and mutation rate file(s):',flush=True)
    for i in range(num_files):
        print(f'\tmhs={files_paths[i]};\n\t\tmutation_map={files_paths_M[i]}')
        mhs_files_M_file[files_paths[i]] = files_paths_M[i]
else:
    num_files_M = 0
    print(f'\nLoaded mhs file(s):',flush=True)
    for i in range(num_files):  
        print(f'\t{files_paths[i]}')
        mhs_files_M_file[files_paths[i]] = 'null'



mhs_files_R_file = {}
if num_files_R>0:
    if num_files!=num_files_R:
        print(f'\tProblem. Recomb rate files provided, but number of files={num_files_R} does not match number of mhs files={num_files}. Aborting',flush=True)
        sys.exit()
    files_paths_r = [file for file in args.input_file_path_R]
    print(f'\nLoaded mhs and recombination rate file(s):',flush=True)
    for i in range(num_files):
        print(f'\tmhs={files_paths[i]};\n\t\trecombination_map={files_paths_r[i]}',flush=True)
        mhs_files_R_file[files_paths[i]] = files_paths_r[i]
else:
    num_files_R = 0
    for i in range(num_files):  
        # print(f'\t{files_paths[i]}')
        mhs_files_R_file[files_paths[i]] = 'null'

# set output path
if args.file_outpath is None:
    output_path = os.getcwd()
    print(f'\nSaving output to {output_path}',flush=True)
    try:
        os.mkdir(output_path)
    except:
        print('\tDirectory already exists.',flush=True)
else:
    output_path = args.file_outpath
    print(f'\nSaving output to {output_path}',flush=True)
output_R_path = args.output_R_path

# set number of states
D = args.number_time_windows
spread_1 = args.D_spread_1
spread_2 = args.D_spread_2
print(f'\nParameters:',flush=True)
print(f'\tnumber of time windows={D}',flush=True)
print(f'\tspread_1={spread_1}; spread_2={spread_2}',flush=True)
lambda_lwr_bnd = args.lambda_lwr_bnd
lambda_upr_bnd = args.lambda_upr_bnd
print(f'\tlambda_lwr_bnds is {lambda_lwr_bnd}',flush=True)
print(f'\tlambda_upr_bnds is {lambda_upr_bnd}',flush=True)
gamma_lwr_bnd = args.gamma_lwr_bnd
gamma_upr_bnd = args.gamma_upr_bnd
print(f'\tgamma_lwr_bnds is {gamma_lwr_bnd}')
print(f'\tgamma_upr_bnds is {gamma_upr_bnd}')

# set bin_size
bin_size = args.bin_size
print(f'\tbin size is {bin_size}',flush=True)

xtol = args.xtol
# print(f'\txtol is {xtol}')

ftol = args.ftol
# print(f'\tftol is {ftol}')

optimisation_method = args.optimisation_method
# print(f'\toptimisation_method is {optimisation_method}')


midpoint_transitions = args.midpoint_transitions
if midpoint_transitions=="False":
    midpoint_transitions = False
else:
    midpoint_transitions = True
print(f'\tmidpoint_transitions is {midpoint_transitions}',flush=True)


midpoint_emissions = args.midpoint_emissions
if midpoint_emissions=="False":
    midpoint_emissions = False
elif midpoint_emissions=="True":
    midpoint_emissions = True
# print(f'\tmidpoint_emissions is {midpoint_emissions}')

final_T_factor = args.final_T_factor
if final_T_factor=="False":
    final_T_factor = None
else:
    final_T_factor = float(final_T_factor)
# print(f'\tfinal_T_factor is {final_T_factor}')

cores = args.cores
if cores==None:
    cores=num_files

# if cores=="False"
# set jump_size
# jump_size = args.jump_size
# print(f'jump_size is {jump_size}')

if args.rho_fixed is False:
    estimate_rho = True
elif args.rho_fixed is True:
    estimate_rho = False

# set index for (T_S,T_E)
T_S_input = args.T_S_index
T_E_input = args.T_E_index
if T_S_input == 'None':
        if T_E_input != 'None':
                print(f'Problem! Input T_S_input is None, therefore T_E_input should be too. Aborting')
        T_S_input = None
        T_E_input = None
        STRUCTURE = False
        PANMIXIA = True
        print('\tinference type: panmictic',flush=True,end='')
        lambda_lwr_bnd_struct = args.lambda_lwr_bnd_struct
        lambda_upr_bnd_struct = args.lambda_upr_bnd_struct
else:
    lambda_lwr_bnd_struct = args.lambda_lwr_bnd_struct
    lambda_upr_bnd_struct = args.lambda_upr_bnd_struct
    print(f'\tlambda_lwr_bnds_struct is {lambda_lwr_bnd_struct}')
    print(f'\tlambda_upr_bnds_struct is {lambda_upr_bnd_struct}')
    try:
            T_S_input = int(T_S_input)
            T_E_input = int(T_E_input)
            STRUCTURE = True
            PANMIXIA = False
            print('\tinference type: structured',flush=True,end='')
    except:
            print(f'Failed converting input T_S and T_E to int. Aborting',flush=True)
print()


# set BW_thresh
if args.BW_threshold is not None:
    BW_thresh = float(args.BW_threshold)
    # print(f'\tBW_thresh is {BW_thresh}')
    print(f'\tthreshold for change in log-likelihood in EM algorithm: {BW_thresh}',flush=True)

else:
    BW_thresh = None
    # print(f'\tBW_thresh is None')

# set BW_its
BW_its = int(args.BW_its)
print(f'\tnumber of iterations in EM algorithm: {BW_its}',flush=True)

save_iteration_files = args.save_iteration_files

# if both BW_its and BW_thresh are defined
if BW_thresh and BW_its:
    print(f'\t\twill iterate until either criteria is met',flush=True)


print(f'\nSequence information:',flush=True)
sequences_info = Parallel(n_jobs=cores, backend='loky')(delayed(bin_sequence)(in_path,bin_size,mhs_files_M_file,mhs_files_R_file) for in_path in files_paths) # returns for mhs file:  het_data, mask_data, j_max, seq_length, num_hets, num_masks, M_sequence_binned, M_vals, R_sequence_binned, R_vals
# zsequences_info = bin_sequence(files_paths[0],bin_size,mhs_files_M_file) # returns for mhs file:  het_data, mask_data, j_max, seq_length, num_hets, num_masks, M_sequence_binned, M_vals
print(f'\t\tFinished getting sequence information.',flush=True)
total_seq_length = sum([sequences_info[i][3] for i in range(0,num_files)])
total_num_hets = sum([sequences_info[i][4] for i in range(0,num_files)])
total_num_masks = sum([sequences_info[i][5] for i in range(0,num_files)])
print(f'\ttotal number of SNPs is {total_num_hets}',flush=True)
print(f'\ttotal sequence_length is {total_seq_length}',flush=True)
print(f'\ttotal number of masks is {total_num_masks}',flush=True)
print(f'\t\tthus of called sites is {total_seq_length - total_num_masks}',flush=True)
j_max = max([sequences_info[i][2] for i in range(num_files)])

sequences_info = downsample_r(sequences_info,num_files_R,args.number_downsamples_R,num_files)

# get mutation and recombination rate
if args.scaled_mutation_rate is None or args.scaled_mutation_rate=='empirical' :
    # print(f'\n\tNo scaled mutation rate given. Using (number of hets)/(callable sequence length)')
    theta = total_num_hets/(total_seq_length - total_num_masks)
    # theta = len(hets)/(true_seq_length - len(masks))
else:
    try:
        theta = float(args.scaled_mutation_rate)
    except:
        print('\t\ttheta input not valid. Aborting',flush=True)
        sys.exit()

if args.mu_over_rho_ratio==None:
    rho = args.scaled_recombination_rate
else:
    try:
        rho = theta / args.mu_over_rho_ratio
    except:
        # print(f'\t "rho = theta / args.mu_over_rho_ratio" failed. Setting rho=theta*0.8')
        rho = theta*0.8


print(f'\tThe scaled mutation rate (theta=4*N*mu) is {theta}',flush=True)
print(f'\tThe scaled recombination rate (rho=4*N*r) is {rho}',flush=True)

# adjust recomb rate for binning
rho = rho*bin_size

# get time boundaries
T = time_intervals(D,spread_1,spread_2,final_T_factor=final_T_factor)
print(f'\tT (coalescent time) is [{",".join([str(i) for i in T])}]') # TODO Write this to file



# emission probabilities 
E = write_emission_probs(D,bin_size,theta,j_max,T,midpoint_end=midpoint_emissions)
E_masked = write_emission_masked_probs(D,bin_size,theta,j_max,T,midpoint_end=midpoint_emissions) 

# set first guess for parameters 
lambda_A_segs = write_segments(args.lambda_A_segments,D)
lambda_A_values = parse_lambda_fg(args.lambda_A_fg,lambda_A_segs)
lambda_A = parse_lambda_input(lambda_A_values,D,lambda_A_segs)

print('\nEM parameters:',flush=True)
print(f'\tfirst guess for lambda_A is [{",".join([str(i) for i in lambda_A])}]',flush=True) # TODO Write this to file
print(f'\ttime segment pattern for lambda_A is {lambda_A_segs}',flush=True)

if PANMIXIA:
    T_S_fg = None
    T_E_fg = None
    gamma_fg = None
    lambda_B_segs = None
    lambda_B_values = None

    # params_first_guess = np.ones(D)
    
elif STRUCTURE:
    gamma_fg = args.gamma_fg
    print(f'\tfirst guess for gamma is {gamma_fg}',flush=True)
    if args.lambda_B_segments is None:
        lambda_B_segs_string = f"{T_S_input}*0,{T_E_input-T_S_input}*1,{D-T_E_input}*0"
    else:
        lambda_B_segs_string = args.lambda_B_segments
    lambda_B_segs = write_segments(lambda_B_segs_string,D)
    if args.lambda_B_fg is None:
        # lambda_B_fg = T_S_input*"-1," + (T_E_input-T_S_input)*"1," + (D-T_E_input-1)*"-1,"+"-1"
        # lambda_B_fg = "-1," + (T_E_input-T_S_input)*"1," + "-1"
        lambda_B_fg = ''
        for i in lambda_B_segs:
            if i[1]==0:
                # lambda_B_fg += f'{i[0]}*0,'
                lambda_B_fg += '1,'
            else:
                lambda_B_fg += i[0]*'1,'
        lambda_B_fg = lambda_B_fg[0:-1]

        # lambda_B_fg = (len(lambda_B_segs)-1)*"1," + "1"
        lambda_B_values = parse_lambda_fg(lambda_B_fg,lambda_B_segs)
    else:
        lambda_B_values = parse_lambda_fg(args.lambda_B_fg,lambda_B_segs)
    lambda_B = parse_lambda_input(lambda_B_values,D,lambda_B_segs)

    print(f'\tfirst guess for lambda_B is [{",".join([str(i) for i in lambda_B])}]')
    print(f'\ttime segment pattern for lambda_B is {lambda_B_segs}')
    print(f'\t\tRecall lambda_B is only valid between indices ts={T_S_input} and te={T_E_input}')

    T_S_fg = T_S_input
    T_E_fg = T_E_input

if inference_flag==True:
    print(f'\nStarting EM algorithm.',flush=True)
    BW = BaumWelch(sequences_info=sequences_info,D=D,E=E,E_masked=E_masked,lambda_A_values=lambda_A_values,lambda_B_values=lambda_B_values,gamma_fg=gamma_fg,lambda_A_segs = lambda_A_segs,lambda_B_segs = lambda_B_segs,rho=rho,theta=theta,estimate_rho=estimate_rho,final_T_factor=final_T_factor,T_array=T,bin_size=bin_size,T_S=T_S_input,T_E=T_E_input,j_max=j_max,spread_1=spread_1,spread_2=spread_2,lambda_lwr_bnd=lambda_lwr_bnd,lambda_upr_bnd=lambda_upr_bnd,gamma_lwr_bnd=gamma_lwr_bnd,gamma_upr_bnd=gamma_upr_bnd,output_path=output_path,cores=cores,xtol=xtol,ftol=ftol,midpoint_transitions=midpoint_transitions,midpoint_end=midpoint_emissions,optimisation_method=optimisation_method,save_iteration_files=save_iteration_files,lambda_lwr_bnd_struct = lambda_lwr_bnd_struct, lambda_upr_bnd_struct = lambda_upr_bnd_struct,recombnoexp=recombnoexp)
    
    BW.BaumWelch(BW_iterations=BW_its,BW_thresh=BW_thresh)
    print(f'\nFinished EM algorithm.',flush=True)
    print(f'\nGetting log_likelihood.',flush=True)
    get_loglikelihood(BW,output_path=output_path)
elif decode_flag==True:
    BW = BaumWelch(sequences_info=sequences_info,D=D,E=E,E_masked=E_masked,lambda_A_values=lambda_A_values,lambda_B_values=lambda_B_values,gamma_fg=gamma_fg,lambda_A_segs = lambda_A_segs,lambda_B_segs = lambda_B_segs,rho=rho,theta=theta,estimate_rho=estimate_rho,final_T_factor=final_T_factor,T_array=T,bin_size=bin_size,T_S=T_S_input,T_E=T_E_input,j_max=j_max,spread_1=spread_1,spread_2=spread_2,lambda_lwr_bnd=lambda_lwr_bnd,lambda_upr_bnd=lambda_upr_bnd,gamma_lwr_bnd=gamma_lwr_bnd,gamma_upr_bnd=gamma_upr_bnd,output_path=output_path,cores=cores,xtol=xtol,ftol=ftol,midpoint_transitions=midpoint_transitions,midpoint_end=midpoint_emissions,optimisation_method=optimisation_method,save_iteration_files=save_iteration_files,lambda_lwr_bnd_struct = lambda_lwr_bnd_struct, lambda_upr_bnd_struct = lambda_upr_bnd_struct,recombnoexp=recombnoexp)
    get_posterior(BW,downsample,output_path,output_R_path)  
else:
    print('\tError!',flush=True)

# deleteme
