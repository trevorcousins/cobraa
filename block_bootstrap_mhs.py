import argparse
import pdb
import pandas as pd
import numpy as np
import sys


def write_mhs(chromstr,pos,sspss,gt,filename):
    # pos is index of hets
    # chrom is int
    current_chr = chromstr
    chr_label = [current_chr]*len(pos)

    with open(filename,'w') as f:
        lis=[chr_label,pos,sspss,gt]
        for x in zip(*lis):
            f.write("{0}\t{1}\t{2}\t{3}\n".format(*x))
    print(f'\t\twritten mhs file to {filename}')
    return None

def calculate_heterozygosity(mhs_array_np,seq_length):
    num_hets = mhs_array_np.shape[0]
    num_masked_sites = np.sum(mhs_array_np[1:,0] - mhs_array_np[0:-1,0] - mhs_array_np[1:,1])
    heterozygosity = num_hets / (seq_length - num_masked_sites)
    return heterozygosity

parser = argparse.ArgumentParser(description="Set parameters for simulation")
parser.add_argument('-inmhs','--inmhs',help='path to input mhs',required=True,type=str)
parser.add_argument('-windowsize','--windowsize',help='windowsize to break mhs in to (scientific notation)',required=True,type=str)
parser.add_argument('-outmhs','--outmhs',help='output path for new mhs',required=True,type=str)
parser.add_argument('-start_block','--start_block',help='Index for starting block (masking blend index)',required=False,type=int,default=1000)



args = parser.parse_args()
zargs = dir(args)
zargs = [zarg for zarg in zargs if zarg[0]!='_']
for zarg in zargs:
    print(f'{zarg} is ',end='')
    exec(f'{zarg}=args.{zarg}')
    exec(f'print(args.{zarg})')

windowsize = int(float(windowsize))
data = pd.read_csv(inmhs, header = None,sep='\t') # load data
data_np = np.array(data.loc[:,1:2])
hets = np.array(data[1]) # read hets position, 0 indexed
hets_diffs = [hets[i+1]-hets[i] for i in range(0,len(hets)-1)]
seq_length = hets[-1] + int(np.mean(hets_diffs)) 
print(f'heterozygosity of {inmhs} = {calculate_heterozygosity(data_np,seq_length)}')

num_windows = int(seq_length/windowsize)
print(f'seq_length={seq_length}')
print(f'num_windows={num_windows}')

block_indices_to_use = np.random.randint(1,num_windows,num_windows)
blocks = {} # keys are indices; values are the blocks

cc=0
num_hets = 0
for i in range(0,len(block_indices_to_use)):
    start = block_indices_to_use[i]*windowsize
    end = (block_indices_to_use[i]+1)*windowsize - 1
    zblock = data_np[(data_np[:,0]>=start) & (data_np[:,0]<end),:]
    blocks[cc] = zblock
    cc+=1
    if zblock.shape[0]>=2:
        num_hets+=zblock.shape[0]

newmhs_np = np.zeros(shape=(num_hets,2),dtype=int)
prev_index = 0
ww = start_block
for i in range(0,len(block_indices_to_use)):
    if blocks[i].shape[0]<2:
        continue
    else:
        zblock = blocks[i]
        diff = zblock[0,0] - ww
        zblock[:,0] = zblock[:,0] - diff
        zblock[0,1] = int(start_block/2)
        newmhs_np[prev_index:prev_index+zblock.shape[0],:] = zblock
        ww = zblock[-1,0] + start_block
        newprevindex = prev_index + zblock.shape[0]
        # print(f'prev_index={prev_index}, new prev_index={newprevindex}')
        # print(f'ww={ww}')
        prev_index = newprevindex

if np.min([newmhs_np[i+1,0]-newmhs_np[i,0] for i in range(0,len(newmhs_np[:,0])-1)])<1:
    print(f'Problem! Min distance between hets is {np.min([newmhs_np[i+1,0]-newmhs_np[i,0] for i in range(0,len(newmhs_np[:,0])-1)])}; it should not be less than one')
    sys.exit()

if np.min(newmhs_np[1:,0] - newmhs_np[0:-1,0] - newmhs_np[1:,1])<0:
    print(f'Problem! Min sspss is {np.min(newmhs_np[1:,0] - newmhs_np[0:-1,0] - newmhs_np[1:,1])}; it should not be less than zero')
    sys.exit()


chromstr = data[0][0]
gt = ['AT']*len(newmhs_np[:,1])
seq_length_new = newmhs_np[-1,0]+1000
print(f'heterozygosity of {outmhs} = {calculate_heterozygosity(newmhs_np,seq_length_new)}')
write_mhs(chromstr,newmhs_np[:,0],newmhs_np[:,1],gt,outmhs)


