import msprime
import numpy as np
import argparse
import pdb
import sys
import os
from structured_transition_matrix_211220 import *
import matplotlib.pyplot as plt
from scipy.special import gamma
import gzip

def get_lineage_data(sim,T_gens,ts,te,verbosity=False):
    # from /home/tc557/ancestry/msprime_simulation_220216_combined.py
    # T_gens is time in gens
    T_s_index = ts # time index of rejoin
    T_e_index = te # time index of split
    nodes_info = np.zeros(shape=(sim.num_nodes,4))
    nodes_info[:,0] = np.arange(0,sim.num_nodes,1)
    nodes_info[:,1] = sim.tables.nodes.flags
    nodes_info[:,2] = sim.tables.nodes.population
    nodes_info[:,3] = sim.tables.nodes.time
    coalesce_path = np.ones(sim.get_num_trees())*(-1) # 0 for A (or AA), 1 for BB, 2 for AB
    # 1048576 is census node
    # 131072 is census 
    
    if verbosity: print(f'\tts_time is T_gens={T_gens[ts]}')
    if verbosity: print(f'\tte_time is T_gens={T_gens[te]}')
    
    i= 0
    for tree in sim.trees():
        if verbosity: print(f'i is {i}')
    #     print(f'\ttmrca is {tree.time(tree.mrca(0,1))}')
        path_0=[]
        v=tree.parent(0)
        while v!=-1:
            path_0.append(v)
            v = tree.parent(v)
        path_1=[]
        w=tree.parent(1)
        while w!= -1:
            path_1.append(w)
            w = tree.parent(w)
    #     print(f'\tparentage of node={0} is {path_0}')
    #     print(f'\tparentage of node={1} is {path_1}')
        census_nodes = []
        
        local_coal_time = tree.time(tree.mrca(0,1))
        if verbosity: print(f'\tlocal_coal_time={local_coal_time}')
#         ()
        current_nodes_label = [u for u in tree.nodes()]
#         ()
        where_census_nodes = np.where(nodes_info[current_nodes_label,1]==1048576)[0]
        if verbosity: print(f'\twhere_census_nodes={where_census_nodes}')
        census_nodes = [current_nodes_label[i] for i in where_census_nodes]
#         census_nodes_new = current_nodes_label[np.where(nodes_info[current_nodes_label,1]==1048576)]
        
#         for u in tree.nodes():
#     #         print(f'node={u},time={tree.time(u)},flag={nodes_info[u,1]}')
#             if nodes_info[u,1]==1048576:
#                 census_nodes.append(u)
    #             print(f'\tnode={u} is census event')
#             if nodes_info[u,3]==tree.time(tree.mrca(0,1)): # tmrca event
    #         if nodes_info[u,1]==0: # coalescent event
#                 coalesce_node = u
    #             print(f'\tnode {u} is a coalescent event')
        
#         print(f'\t i is {i}, census_nodes {census_nodes}')
#         print(f'\t i is {i}, census_nodes_new {census_nodes_new}')
        
#         print(f'\t i is {i}, coalesce_node {coalesce_node}')
#         print(f'\t i is {i}, coalesce_node time is  {tree.time(coalesce_node)}')
#         print(f'\t i is {i}, coalesce time {tree.time(tree.mrca(0,1))} ' )
#         print(f'\n')
        
    #     print(f'\t***********************')
    #     print(f'\tcensus nodes are {census_nodes}')
        if local_coal_time<T_gens[T_s_index]:        
    #         print('\tCoal before structured period.')
    #         print('\tLineage path is {0}')
            coalesce_path[i] = 0
        elif local_coal_time>=T_gens[T_s_index] and local_coal_time<T_gens[T_e_index]:
    #         print('\tCoalesce in structured period')
    #         print('\tlineage_path unclear to me right now')
#             ()
            cens_path_0 = [cens for cens in census_nodes if cens in path_0]
            cens_path_1 = [cens for cens in census_nodes if cens in path_1]
            if len(cens_path_0)>1 or len(cens_path_1)>1 or len(cens_path_1)==0 or len(cens_path_1)==0:
                print(f'*********************something odd going on. The number of census events is not equal to 1*****************',flush=True)
                print(f'\tline 85 in msprime_Simulation_220216_combined.py',flush=True)
                coalesce_path = np.ones(sim.get_num_trees())*(-1)
                return coalesce_path
            else:
                try:
                    cens_path_0 = cens_path_0[0]
                    cens_path_1 = cens_path_1[0]
                except:
                    print(f'***************census event not found for this node. I dont know why this fails, but will skip this iteration',flush=True)
                    coalesce_path = np.ones(sim.get_num_trees())*(-1)
                    return coalesce_path
            ancestry_path_0 = int(nodes_info[cens_path_0,2])
            ancestry_path_1 = int(nodes_info[cens_path_1,2])
            if verbosity: print(f'\tancestry_path_0 is {ancestry_path_0}')
            if verbosity: print(f'\tancestry_path_1 is {ancestry_path_1}')

    #         print(f'\tnode=0 went through pop {ancestry_path_0}')
    #         print(f'\tnode=1 went through pop {ancestry_path_1}')
    #         print(f'\tLineage path is {ancestry_path_0,ancestry_path_1}')
            if (ancestry_path_0+ancestry_path_1)==0: # path AA
                coalesce_path[i] = 0
            elif (ancestry_path_0+ancestry_path_1)==2: # path BB
                coalesce_path[i] = 1
            elif (ancestry_path_0+ancestry_path_1)==1:
                print(f'*************This should not happen. Should be (0,0), or (1,1)***********',flush=True)

        elif local_coal_time>=T_gens[T_e_index]:
    #         print('\tCoalesce post structured period')
    #         ()
            cens_path_0 = [cens for cens in census_nodes if cens in path_0]
            cens_path_1 = [cens for cens in census_nodes if cens in path_1]
            if len(cens_path_0)>1 or len(cens_path_1)>1 or len(cens_path_1)==0 or len(cens_path_1)==0:
                print(f'*********************something odd going on. The number of census events is not equal to 1*****************',flush=True)
                print(f'\tline 111 in msprime_Simulation_220216_combined.py',flush=True)
                coalesce_path = np.ones(sim.get_num_trees())*(-1)
                return coalesce_path
            else:
                try:
                    cens_path_0 = cens_path_0[0]
                    cens_path_1 = cens_path_1[0]
                except:
                    print(f'***************census event not found for this node. I dont know why this fails, but will skip this iteration',flush=True)
                    coalesce_path = np.ones(sim.get_num_trees())*(-1)
                    return coalesce_path

            ancestry_path_0 = int(nodes_info[cens_path_0,2])
            ancestry_path_1 = int(nodes_info[cens_path_1,2])
            if verbosity: print(f'\tancestry_path_0 is {ancestry_path_0}')
            if verbosity: print(f'\tancestry_path_1 is {ancestry_path_1}')
            
    #         print(f'\tnode=0 went through pop {ancestry_path_0}')
    #         print(f'\tnode=1 went through pop {ancestry_path_1}')
    #         print(f'\tLineage path is {ancestry_path_0,ancestry_path_1}')
            if (ancestry_path_0+ancestry_path_1)==0:
                coalesce_path[i] = 0
            elif (ancestry_path_0+ancestry_path_1)==2:
                coalesce_path[i] = 1
            elif (ancestry_path_0+ancestry_path_1)==1:
                coalesce_path[i] = 2
        if verbosity: print(f'\tcoalesce_path[i] is {coalesce_path[i]}')

        

    #     print(tree.parent(0))
    #     print(tree.nodes(0))
    #     print()
        i+= 1
    return coalesce_path

def combine_coal_data_lineage(coal_data,coalesce_path):
    coal_data_lineage = np.zeros(shape=(coal_data.shape[0],4))
    coal_data_lineage[:,-1] = coalesce_path
    coal_data_lineage[:,0:3] = coal_data
    return coal_data_lineage

def mean_beta(alpha,beta):
    mean = alpha / (alpha + beta)
    print(f'mean of beta is {mean}')
    return mean



def get_path_index(regular_index,D,ts,te):
    # TODO FIX THIS 
    if regular_index>=D:
        print(f'index={regular_index} cannot be bigger than D={D}')
        return None
    te_path = ts + (te-ts)*2
    if regular_index<ts:
        path_index = regular_index
    elif regular_index>=ts and regular_index<te:
        path_index = ts + (regular_index-ts)*2
    elif regular_index>=te:
        path_index = te_path + (regular_index-te)*3
    else:
        print(f'mistake in get_path_index for row_path: row is {row}')
    return path_index

def get_coal_data(sim):
    tmrca_data = [] # first column is index, second column is span of tree, third column is tMRCA
    tmrca_data = np.zeros(shape=(sim.get_num_trees(),3))
    index = 0
    for tree in sim.trees():
        # ()
    #     print(f"tree_index={tree.index} \ntree_rinterval={tree.interval[1]} \ntree_TMRCA={tree.time(tree.parent(0))} \n")
        tmrca_data[index,0] = tree.index
        tmrca_data[index,1] = tree.interval[1]
        tmrca_data[index,2] = tree.time(tree.mrca(0,1))
        # print("tree {}: interval = {}".format(tree.index, tree.interval))
        # print("TMRCA: {}".format(tree.time(tree.mrca(0,1))))

        # print(tree.draw(format="unicode"))
        index += 1
    return tmrca_data

def generate_random_mutation_map(mu_mean,chrom_length,change_rate,alpha,beta,num_mu_domains):
    num_changes = int(chrom_length/change_rate)
    change_points = np.random.exponential(change_rate,size=num_changes)
    # mu_domain = np.array([mu_mean*(1.1**i) for i in range(-int(num_mu_domains/2),int(num_mu_domains/2))])
    # x_domain = np.linspace(0,1,num_mu_domains)
    # beta_pdf_array = beta_pdf(x_domain,alpha,beta)
    # beta_pdf_array_normalised = beta_pdf(x_domain,alpha,beta)/np.sum(beta_pdf_array)
    # mus = np.random.choice(mu_domain, size=num_changes-1, replace=True, p=beta_pdf_array_normalised)
    mean_beta_value = mean_beta(alpha,beta)
    transform = mean_beta_value/mu_mean
    mus = np.random.beta(alpha,beta,num_changes-1)/transform

    print(f'num_changes is {num_changes}')
    print(f'min(mus)={min(mus)}; max(mus)={max(mus)}')
    print(f'mean(mus)={np.mean(mus)}')
    position = np.zeros(shape=change_points.shape)
    for i in range(0,num_changes):
        position[i] = np.sum(change_points[0:i])
    return position, mus

def beta_pdf(x,a,b):
    B = (gamma(a)*gamma(b))/gamma(a+b)
    pdf_numerator = (x**(a-1))*((1-x)**(b-1))
    pdf = pdf_numerator/B
    return pdf


def const_pop220110(N,mu,p,L):
    # constant population size of diploid size N
	# N diploid size
	# L sequence length
	# mu per gen per bp mutation rate
	# p per gen per bp recombination rate
	

    population_configurations = [
        msprime.PopulationConfiguration(
            sample_size=2, initial_size=N, growth_rate=0),
			]
    demographic_events = [
    ]
    # Use the demography debugger to print out the demographic history
    # that we have just described.
    dd = msprime.DemographyDebugger(
        population_configurations=population_configurations,demographic_events=demographic_events)
    print('Demographic history:\n')
    dd.print_history()
    # print('Starting simulation')
    sim = msprime.simulate(population_configurations=population_configurations,
                           demographic_events=demographic_events, length=L, recombination_rate=p,
                           mutation_rate=mu,record_full_arg=True)
    # print('Finished simulation')
    return sim

def pop_expansion220110(N,mu,p,L,T,start_index,end_index,intensity,simulation_model="hudson"):
    # population of diploid size N that expands to size N*intensity at time T[start_index]
    # and return to size N at time T[end_index]
	# N diploid size
	# L sequence length
	# mu per gen per bp mutation rate
	# p per gen per bp recombination rate
	# T time interval boundaries, IN GENERATIONS.
    # start_index index in T where change starts
    # end_index index in T where change stops

    print(f'\t\tstart_index is {start_index}')
    print(f'\t\tend_index is {end_index}')
    print(f'\t\tintensity is {intensity}')
    print(f'\t\tsimulation model is {simulation_model}')


    population_configurations = [
        msprime.PopulationConfiguration(
            sample_size=2, initial_size=N, growth_rate=0),
			]
    demographic_events = [
        msprime.PopulationParametersChange(time=T[start_index], initial_size=N*intensity),
        msprime.PopulationParametersChange(time=T[end_index], initial_size=N),

    ]
    # Use the demography debugger to print out the demographic history
    # that we have just described.
    dd = msprime.DemographyDebugger(
        population_configurations=population_configurations,demographic_events=demographic_events)
    print('Demographic history:\n')
    dd.print_history()
    # print('Starting simulation')
    sim = msprime.simulate(population_configurations=population_configurations,
                           demographic_events=demographic_events, length=L, recombination_rate=p,
                           mutation_rate=mu,record_full_arg=True,model=simulation_model)
    # print('Finished simulation')
    return sim

def pop_split_220113(N_A,N_B,mu,p,L,T,start_index,end_index,gamma,simulation_model="hudson"):
    

    print(f'\t\tstart_index is {start_index}')
    print(f'\t\tend_index is {end_index}')
    print(f'\t\tgamma is {gamma}')
    print(f'\t\tsimulation model is {simulation_model}')


    population_configurations = [
        msprime.PopulationConfiguration(
            sample_size=2, initial_size=N_A, growth_rate=0),
        msprime.PopulationConfiguration(
            sample_size=0, initial_size=N_B, growth_rate=0),
    ]
    migration_matrix = [[0,0],[0,0]]
    demographic_events = [
        msprime.MassMigration(
            time=T[start_index], source = 0, dest=1,proportion=gamma),
        msprime.MassMigration(
            time=T[end_index], source=1, dest=0, proportion=1)
    ]
    # Use the demography debugger to print out the demographic history
    # that we have just described.
    dd = msprime.DemographyDebugger(
        population_configurations=population_configurations,
        migration_matrix=migration_matrix,
        demographic_events=demographic_events)
    dd.print_history()
    sim = msprime.simulate(population_configurations=population_configurations,
                           migration_matrix=migration_matrix,
                           demographic_events=demographic_events, length=L, recombination_rate=p,
                           mutation_rate=mu,record_full_arg=True,model=simulation_model)
    return sim


def pop_split_220114(N_A,N_B,mu,p,L,T,start_index,end_index,gamma,simulation_model="hudson",return_debug=False):
    
    print(f'\t\tstart_index is {start_index}')
    print(f'\t\tend_index is {end_index}')
    print(f'\t\tgamma is {gamma}')
    print(f'\t\tsimulation model is {simulation_model}')



    demography = msprime.Demography()
    demography.add_population(name="A", initial_size=N)
    demography.add_population(name="B", initial_size=N)
    demography.add_mass_migration(time=T[start_index], source="A", dest="B", proportion=gamma)
    demography.add_mass_migration(time=T[end_index], source="B", dest="A", proportion=1)

    debug = demography.debug()
    print(debug)
    print('Starting simulation')
    sim = msprime.sim_ancestry(samples={"A": 1, "B": 0},ploidy=2,demography=demography,sequence_length=L,recombination_rate=p,record_full_arg=True,model=simulation_model)
    msim = msprime.sim_mutations(sim, rate=mu)
    print('Finished simulation')
    # can return debug too for coalescence_rate_trajectory and stuff
    if return_debug is False:
        return msim
    else:
        return msim, debug
def matching_psc_pop_split_220120(N_A,N_B,mu,p,L,T,start_index,end_index,gamma,pop_size_changes,simulation_model="hudson"):
    
    print(f'\t\tSimulating panmictic population with 1/CR equal to split model with following parameters:')
    print(f'\t\tstart_index is {start_index}')
    print(f'\t\tend_index is {end_index}')
    print(f'\t\tgamma is {gamma}')
    print(f'\t\tsimulation model is {simulation_model}')


    demography = msprime.Demography()
    demography.add_population(name="A", initial_size=N)
    for i in range(start_index,end_index+1):
        demography.add_population_parameters_change(T[i], initial_size=pop_size_changes[i], growth_rate=None, population="A")
        
    debug = demography.debug()
    print(debug)
    print('Starting simulation')
    sim = msprime.sim_ancestry(samples={"A": 1},ploidy=2,demography=demography,sequence_length=L,recombination_rate=p,record_full_arg=True,model=simulation_model)
    msim = msprime.sim_mutations(sim, rate=mu)
    print('Finished simulation')
    return msim, debug

def pop_split_psc_pop_220119(N,mu,p,L,T,start_index_str,end_index_str,start_index_psc,end_index_psc,gamma,intensity,pop_to_change,simulation_model="hudson",verbosity=False):
    
    print(f'\t\tstart_index_str is {start_index_str}')
    print(f'\t\tend_index_str is {end_index_str}')
    print(f'\t\tstart_index_psc is {start_index_psc}')
    print(f'\t\tend_index_psc is {end_index_psc}')
    print(f'\t\tgamma is {gamma}')
    print(f'\t\tintensity is {intensity}')
    print(f'\t\tpop_to_change is {pop_to_change}')

    print(f'\t\tsimulation model is {simulation_model}')


    demography = msprime.Demography()
    demography.add_population(name="A", initial_size=N)
    demography.add_population(name="B", initial_size=N)
    demography.add_population_parameters_change(T[start_index_psc], initial_size=N*intensity, growth_rate=None, population=pop_to_change)
    demography.add_population_parameters_change(T[end_index_psc], initial_size=N, growth_rate=None, population=pop_to_change)
    demography.add_mass_migration(time=T[start_index_str], source="A", dest="B", proportion=gamma)
    demography.add_mass_migration(time=T[end_index_str], source="B", dest="A", proportion=1)
    demography.sort_events()
    debug = demography.debug()
    if verbosity==True:
        print(debug)
    print('Starting simulation')
    sim = msprime.sim_ancestry(samples={"A": 1, "B": 0},ploidy=2,demography=demography,sequence_length=L,recombination_rate=p,record_full_arg=True,model=simulation_model)
    msim = msprime.sim_mutations(sim, rate=mu)
    print('Finished simulation')
    return msim, debug

def pop_split_psc_pop_220207(N,mu,p,L,T,start_index_str,end_index_str,gamma,lambda_array,pop_to_change,simulation_model="hudson",verbosity=False):
    
    demography = msprime.Demography()
    demography.add_population(name="A", initial_size=N)
    demography.add_population(name="B", initial_size=N)
    
    for i in range(0,len(lambda_array)):
        demography.add_population_parameters_change(T[i], initial_size=N*(1/lambda_array[i]), growth_rate=None, population='A')
    demography.add_mass_migration(time=T[start_index_str], source="A", dest="B", proportion=gamma)
    demography.add_mass_migration(time=T[end_index_str], source="B", dest="A", proportion=1)

    demography.sort_events()
    debug = demography.debug()
    if verbosity: print(debug)
    print('Starting simulation')
    sim = msprime.sim_ancestry(samples={"A": 1, "B": 0},ploidy=2,demography=demography,sequence_length=L,recombination_rate=p,record_full_arg=True,model=simulation_model)
    msim = msprime.sim_mutations(sim, rate=mu)
    print('Finished simulation')
    return msim, debug

def pop_split_model_220913(N,lambda_array,mu,gamma,p,L,T,start_index,end_index,simulation_model="hudson",return_debug=False,recordfullarg=False,verbosity=True):
    # T in gens
    # I use N as haploid here

    demography = msprime.Demography()
    demography.add_population(name="A", initial_size=N)
    demography.add_population(name="B", initial_size=N)
    demography.add_mass_migration(time=T[start_index], source="A", dest="B", proportion=gamma)
    demography.add_mass_migration(time=T[end_index], source="B", dest="A", proportion=1)
    for i in range(0,len(lambda_array)):
        demography.add_population_parameters_change(T[i], initial_size=(N )/lambda_array[i], growth_rate=None, population='A')
    demography.sort_events()
    debug = demography.debug()
    if verbosity: print(debug)
    print('Starting simulation')
    sim = msprime.sim_ancestry(samples={"A": 1, "B": 0},ploidy=2,demography=demography,sequence_length=L,recombination_rate=p,record_full_arg=recordfullarg,model=simulation_model)
    msim = msprime.sim_mutations(sim, rate=mu)
    print('Finished simulation')
    # can return debug too for coalescence_rate_trajectory and stuff
    if return_debug is False:
        return msim
    else:
        return msim, debug

def pop_cont_model_220913(N,lambda_array,mu,m,p,L,T,start_index,end_index,simulation_model="hudson",return_debug=False,recordfullarg=False,verbosity=True):
    # T in gens
    # I use N as haploid here
    # pdb.set_trace()
    demography = msprime.Demography()
    demography.add_population(name="A", initial_size=N)
    demography.add_population(name="B", initial_size=N)
    print(f'start_index = {start_index}')
    demography.add_symmetric_migration_rate_change(time=T[start_index],populations=['A','B'],rate=m)
    demography.add_symmetric_migration_rate_change(time=T[end_index],populations=['A','B'],rate=0)
    demography.add_mass_migration(time=T[end_index], source="B", dest="A", proportion=1)
    for i in range(0,len(lambda_array)):
        demography.add_population_parameters_change(T[i], initial_size=(N)/lambda_array[i], growth_rate=None, population='A')
    demography.sort_events()
    debug = demography.debug()
    if verbosity: print(debug)
    print('Starting simulation')
    sim = msprime.sim_ancestry(samples={"A": 1, "B": 0},ploidy=2,demography=demography,sequence_length=L,recombination_rate=p,record_full_arg=recordfullarg,model=simulation_model)
    msim = msprime.sim_mutations(sim, rate=mu)
    print('Finished simulation')
    # can return debug too for coalescence_rate_trajectory and stuff
    if return_debug is False:
        return msim
    else:
        return msim, debug


def pop_split_psc_pop_220312(N,mu,p,L,T,start_index_str,end_index_str,start_index_psc,end_index_psc,gamma,intensity,pop_to_change,simulation_model="hudson",verbosity=True):
    
    demography = msprime.Demography()
    demography.add_population(name="A", initial_size=N)
    demography.add_population(name="B", initial_size=N)
    demography.add_population_parameters_change(T[start_index_psc], initial_size=N*intensity, growth_rate=None, population=pop_to_change)
    demography.add_population_parameters_change(T[end_index_psc], initial_size=N, growth_rate=None, population=pop_to_change)
    demography.add_mass_migration(time=T[start_index_str], source="A", dest="B", proportion=gamma)
    demography.add_mass_migration(time=T[end_index_str], source="B", dest="A", proportion=1)
    demography.add_census(time=(T[start_index_str]+1) )
    demography.sort_events()
    debug = demography.debug()
    if verbosity: print(debug)
    print('Starting simulation')
    sim = msprime.sim_ancestry(samples={"A": 1, "B": 0},ploidy=2,demography=demography,sequence_length=L,recombination_rate=p,record_full_arg=True,model=simulation_model)
    msim = msprime.sim_mutations(sim, rate=mu)
    print('Finished simulation')
    return msim, debug

def pop_split_psc_pop_220425(N,mu,p,L,T,start_index_str,end_index_str,start_index_psc,end_index_psc,gamma,intensity,pop_to_change,simulation_model="hudson",verbosity=True):
    D = len(T)
    D = len(T) - 1
    ts=start_index_str
    te=end_index_str
    A_popsize = np.ones(D)
    A_popsize[0:3] = 0.3
    A_popsize[3:6] = 3
    # A_popsize[ts:te] = intensity
    A_popsize[te:te+5] = 0.3
    A_popsize[te+5:te+10] = 3
    B_popsize = np.ones(D)
    exec(f'{pop_to_change}_popsize[ts:te] = intensity')

    print(f'A_popsize is {A_popsize*N}')
    print(f'A_popsize (scaled) is {A_popsize}')
    print(f'B_popsize is {B_popsize*N}')
    print(f'B_popsize (scaled) is {B_popsize}')
    demography = msprime.Demography()
    demography.add_population(name="A", initial_size=N)
    demography.add_population(name="B", initial_size=N)
    for i in range(0,len(T)-1):
        demography.add_population_parameters_change(T[i], initial_size=N*A_popsize[i], growth_rate=None, population='A')
        demography.add_population_parameters_change(T[i], initial_size=N*B_popsize[i], growth_rate=None, population='B')

    demography.add_mass_migration(time=T[start_index_str], source="A", dest="B", proportion=gamma)
    demography.add_mass_migration(time=T[end_index_str], source="B", dest="A", proportion=1)
    demography.add_census(time=(T[start_index_str]+1) )
    demography.sort_events()
    debug = demography.debug()
    if verbosity: print(debug)
    print('Starting simulation')
    sim = msprime.sim_ancestry(samples={"A": 1, "B": 0},ploidy=2,demography=demography,sequence_length=L,recombination_rate=p,record_full_arg=True,model=simulation_model)
    # msim = msprime.sim_mutations(sim, rate=mu)
    print('Finished simulation')
    # return msim, debug
    return sim, debug


def pop_split_custom_lambda_220721(N,lambdas_A,lambdas_B,mu,p,L,T,start_index_str,end_index_str,gamma,simulation_model="hudson",verbosity=True,recordfullarg=True):
    # T in gens
    print(f'N is {N}')
    print(f'lambdas_A is {lambdas_A}')
    print(f'lambdas_B is {lambdas_B}')

    print(f'mu is {mu}')
    print(f'p is {p}')
    print(f'L is {L}')
    print(f'T is {T}')
    print(f'ts is {start_index}')
    print(f'te is {end_index}')
    print(f'gamma is {gamma}')

    ts=start_index_str
    te=end_index_str
    A_popsize = N / lambdas_A
    B_popsize = N / lambdas_B

    print(f'A_popsize is {A_popsize}')
    print(f'B_popsize is {B_popsize}')
    demography = msprime.Demography()
    demography.add_population(name="A", initial_size=N)
    demography.add_population(name="B", initial_size=N)
    for i in range(0,len(T)-1):
        demography.add_population_parameters_change(T[i], initial_size=A_popsize[i], growth_rate=None, population='A')
        demography.add_population_parameters_change(T[i], initial_size=B_popsize[i], growth_rate=None, population='B')

    demography.add_mass_migration(time=T[ts], source="A", dest="B", proportion=gamma)
    demography.add_mass_migration(time=T[te], source="B", dest="A", proportion=1)
    demography.add_census(time=(T[ts]+1) )
    demography.sort_events()
    debug = demography.debug()
    if verbosity: print(debug)
    print('Starting simulation')
    sim = msprime.sim_ancestry(samples={"A": 1, "B": 0},ploidy=2,demography=demography,sequence_length=L,recombination_rate=p,record_full_arg=recordfullarg,model=simulation_model)
    # msim = msprime.sim_mutations(sim, rate=mu)
    print('Finished simulation')
    # return msim, debug
    return sim, debug

def pop_custom_lambda_220721(N,lambdas_A,lambdas_B,mu,p,L,T,start_index_str,end_index_str,gamma,simulation_model="hudson",verbosity=True,recordfullarg=True):
    # T in gens
    print(f'N is {N}')
    print(f'lambdas_A is {lambdas_A}')

    print(f'mu is {mu}')
    print(f'p is {p}')
    print(f'L is {L}')
    print(f'T is {T}')
    A_popsize = N / lambdas_A

    print(f'A_popsize is {A_popsize}')
    demography = msprime.Demography()
    demography.add_population(name="A", initial_size=N)
    for i in range(0,len(T)-1):
        demography.add_population_parameters_change(T[i], initial_size=A_popsize[i], growth_rate=None, population='A')
    demography.add_census(time=(T[0]+1) )
    demography.sort_events()
    debug = demography.debug()
    if verbosity: print(debug)
    print('Starting simulation')
    sim = msprime.sim_ancestry(samples={"A": 1},ploidy=2,demography=demography,sequence_length=L,recombination_rate=p,record_full_arg=recordfullarg,model=simulation_model)
    # msim = msprime.sim_mutations(sim, rate=mu)
    print('Finished simulation')
    # return msim, debug
    return sim, debug



def pop_split_psc_pop_220612(N,mu,p,L,T,start_index_str,end_index_str,start_index_psc,end_index_psc,gamma,intensity,pop_to_change,simulation_model="hudson",verbosity=True):
    D = len(T)
    D = len(T) - 1
    ts=start_index_str
    te=end_index_str
    A_popsize = np.ones(D)
    A_popsize[0:3] = 0.3
    A_popsize[3:6] = 3
    # A_popsize[ts:te] = intensity
    A_popsize[te:te+5] = 3
    A_popsize[te+5:te+10] = 0.8
    B_popsize = np.ones(D)
    exec(f'{pop_to_change}_popsize[ts:te] = intensity')

    print(f'A_popsize is {A_popsize*N}')
    print(f'A_popsize (scaled) is {A_popsize}')
    print(f'B_popsize is {B_popsize*N}')
    print(f'B_popsize (scaled) is {B_popsize}')
    demography = msprime.Demography()
    demography.add_population(name="A", initial_size=N)
    demography.add_population(name="B", initial_size=N)
    for i in range(0,len(T)-1):
        demography.add_population_parameters_change(T[i], initial_size=N*A_popsize[i], growth_rate=None, population='A')
        demography.add_population_parameters_change(T[i], initial_size=N*B_popsize[i], growth_rate=None, population='B')

    demography.add_mass_migration(time=T[start_index_str], source="A", dest="B", proportion=gamma)
    demography.add_mass_migration(time=T[end_index_str], source="B", dest="A", proportion=1)
    demography.add_census(time=(T[start_index_str]+1) )
    demography.sort_events()
    debug = demography.debug()
    if verbosity: print(debug)
    print('Starting simulation')
    sim = msprime.sim_ancestry(samples={"A": 1, "B": 0},ploidy=2,demography=demography,sequence_length=L,recombination_rate=p,record_full_arg=True,model=simulation_model)
    # msim = msprime.sim_mutations(sim, rate=mu)
    print('Finished simulation')
    # return msim, debug
    return sim, debug

def pop_split_psc_pop_220613(N,mu,p,L,T,start_index_str,end_index_str,start_index_psc,end_index_psc,gamma,intensity,pop_to_change,simulation_model="hudson",verbosity=True):
    D = len(T)
    D = len(T) - 1
    ts=start_index_str
    te=end_index_str
    A_popsize = np.ones(D)
    A_popsize[0:3] = 0.3
    A_popsize[3:6] = 3
    # A_popsize[ts:te] = intensity
    A_popsize[te:te+5] = 1.2
    A_popsize[te+5:te+10] = 0.8
    B_popsize = np.ones(D)
    exec(f'{pop_to_change}_popsize[ts:te] = intensity')

    print(f'A_popsize is {A_popsize*N}')
    print(f'A_popsize (scaled) is {A_popsize}')
    print(f'B_popsize is {B_popsize*N}')
    print(f'B_popsize (scaled) is {B_popsize}')
    demography = msprime.Demography()
    demography.add_population(name="A", initial_size=N)
    demography.add_population(name="B", initial_size=N)
    for i in range(0,len(T)-1):
        demography.add_population_parameters_change(T[i], initial_size=N*A_popsize[i], growth_rate=None, population='A')
        demography.add_population_parameters_change(T[i], initial_size=N*B_popsize[i], growth_rate=None, population='B')

    demography.add_mass_migration(time=T[start_index_str], source="A", dest="B", proportion=gamma)
    demography.add_mass_migration(time=T[end_index_str], source="B", dest="A", proportion=1)
    demography.add_census(time=(T[start_index_str]+1) )
    demography.sort_events()
    debug = demography.debug()
    if verbosity: print(debug)
    print('Starting simulation')
    sim = msprime.sim_ancestry(samples={"A": 1, "B": 0},ploidy=2,demography=demography,sequence_length=L,recombination_rate=p,record_full_arg=True,model=simulation_model)
    # msim = msprime.sim_mutations(sim, rate=mu)
    print('Finished simulation')
    # return msim, debug
    return sim, debug

def pop_split_psc_pop_220614(N,mu,p,L,T,start_index_str,end_index_str,start_index_psc,end_index_psc,gamma,intensity,pop_to_change,simulation_model="hudson",verbosity=True):
    D = len(T)
    D = len(T) - 1
    ts=start_index_str
    te=end_index_str
    A_popsize = np.ones(D)
    A_popsize[0:3] = 0.3
    A_popsize[3:6] = 6
    # A_popsize[ts:te] = intensity
    A_popsize[te+5:te+10] = 8
    A_popsize[te+10:te+4] = 0.125
    B_popsize = np.ones(D)
    exec(f'{pop_to_change}_popsize[ts:te] = intensity')

    print(f'A_popsize is {A_popsize*N}')
    print(f'A_popsize (scaled) is {A_popsize}')
    print(f'B_popsize is {B_popsize*N}')
    print(f'B_popsize (scaled) is {B_popsize}')
    demography = msprime.Demography()
    demography.add_population(name="A", initial_size=N)
    demography.add_population(name="B", initial_size=N)
    for i in range(0,len(T)-1):
        demography.add_population_parameters_change(T[i], initial_size=N*A_popsize[i], growth_rate=None, population='A')
        demography.add_population_parameters_change(T[i], initial_size=N*B_popsize[i], growth_rate=None, population='B')

    demography.add_mass_migration(time=T[start_index_str], source="A", dest="B", proportion=gamma)
    demography.add_mass_migration(time=T[end_index_str], source="B", dest="A", proportion=1)
    demography.add_census(time=(T[start_index_str]+1) )
    demography.sort_events()
    debug = demography.debug()
    if verbosity: print(debug)
    print('Starting simulation')
    sim = msprime.sim_ancestry(samples={"A": 1, "B": 0},ploidy=2,demography=demography,sequence_length=L,recombination_rate=p,record_full_arg=True,model=simulation_model)
    # msim = msprime.sim_mutations(sim, rate=mu)
    print('Finished simulation')
    # return msim, debug
    return sim, debug


def pop_split_psc_pop_b220614(N,mu,p,L,T,start_index_str,end_index_str,start_index_psc,end_index_psc,gamma,intensity,pop_to_change,simulation_model="hudson",verbosity=True):
    D = len(T)
    D = len(T) - 1
    ts=start_index_str
    te=end_index_str
    A_popsize = np.ones(D)
    A_popsize[0:3] = 0.3
    A_popsize[3:6] = 6
    # A_popsize[ts:te] = intensity
    A_popsize[te+5:te+10] = 0.125
    A_popsize[te+10:te+4] = 8
    B_popsize = np.ones(D)
    exec(f'{pop_to_change}_popsize[ts:te] = intensity')

    print(f'A_popsize is {A_popsize*N}')
    print(f'A_popsize (scaled) is {A_popsize}')
    print(f'B_popsize is {B_popsize*N}')
    print(f'B_popsize (scaled) is {B_popsize}')
    demography = msprime.Demography()
    demography.add_population(name="A", initial_size=N)
    demography.add_population(name="B", initial_size=N)
    for i in range(0,len(T)-1):
        demography.add_population_parameters_change(T[i], initial_size=N*A_popsize[i], growth_rate=None, population='A')
        demography.add_population_parameters_change(T[i], initial_size=N*B_popsize[i], growth_rate=None, population='B')

    demography.add_mass_migration(time=T[start_index_str], source="A", dest="B", proportion=gamma)
    demography.add_mass_migration(time=T[end_index_str], source="B", dest="A", proportion=1)
    demography.add_census(time=(T[start_index_str]+1) )
    demography.sort_events()
    debug = demography.debug()
    if verbosity: print(debug)
    print('Starting simulation')
    sim = msprime.sim_ancestry(samples={"A": 1, "B": 0},ploidy=2,demography=demography,sequence_length=L,recombination_rate=p,record_full_arg=True,model=simulation_model)
    # msim = msprime.sim_mutations(sim, rate=mu)
    print('Finished simulation')
    # return msim, debug
    return sim, debug


def pop_custom_220429a(N,mu,p,L,T,intensity,simulation_model="hudson",verbosity=True):
    D = len(T)
    D = len(T) - 1
    A_popsize = np.ones(D)
    A_popsize[0:3] = 4
    A_popsize[3:6] = 0.1
    A_popsize[6:20] = intensity
    A_popsize[20:25] = 0.2
    A_popsize[25:32] = 2

    print(f'A_popsize is {A_popsize*N}')
    print(f'A_popsize (scaled) is {A_popsize}')
    demography = msprime.Demography()
    demography.add_population(name="A", initial_size=N)
    for i in range(0,len(T)-1):
        demography.add_population_parameters_change(T[i], initial_size=N*A_popsize[i], growth_rate=None, population='A')

    demography.sort_events()
    debug = demography.debug()
    if verbosity: print(debug)
    print('Starting simulation')
    sim = msprime.sim_ancestry(samples={"A": 1},ploidy=2,demography=demography,sequence_length=L,recombination_rate=p,record_full_arg=True,model=simulation_model)
    # msim = msprime.sim_mutations(sim, rate=mu)
    print('Finished simulation')
    # return msim, debug
    return sim, debug


def pop_custom_220429b(N,mu,p,L,T,intensity,simulation_model="hudson",verbosity=True):
    D = len(T)
    D = len(T) - 1
    A_popsize = np.ones(D)
    A_popsize[0:3] = 4
    A_popsize[3:6] = 0.1
    A_popsize[6:20] = intensity
    A_popsize[20:25] = 2
    A_popsize[25:32] = 0.2

    print(f'A_popsize is {A_popsize*N}')
    print(f'A_popsize (scaled) is {A_popsize}')
    demography = msprime.Demography()
    demography.add_population(name="A", initial_size=N)
    for i in range(0,len(T)-1):
        demography.add_population_parameters_change(T[i], initial_size=N*A_popsize[i], growth_rate=None, population='A')

    demography.sort_events()
    debug = demography.debug()
    if verbosity: print(debug)
    print('Starting simulation')
    sim = msprime.sim_ancestry(samples={"A": 1},ploidy=2,demography=demography,sequence_length=L,recombination_rate=p,record_full_arg=True,model=simulation_model)
    # msim = msprime.sim_mutations(sim, rate=mu)
    print('Finished simulation')
    # return msim, debug
    return sim, debug

def psc_europeanlike_220427(N,mu,T,lambda_array,recombination_map,simulation_model="hudson",verbosity=False,recordfullarg=False):
  
    demography = msprime.Demography()
    demography.add_population(name="A", initial_size=N)
    
    for i in range(0,len(lambda_array)):
        demography.add_population_parameters_change(T[i], initial_size=N*(1/lambda_array[i]), growth_rate=None, population='A')
    demography.sort_events()
    debug = demography.debug()
    if verbosity: print(debug)
    print('Starting simulation')
    sim = msprime.sim_ancestry(samples={"A": 1},ploidy=2,demography=demography,sequence_length=recombination_map.right[-1],recombination_rate=recombination_map,record_full_arg=recordfullarg,model=simulation_model)
    msim = msprime.sim_mutations(sim, rate=mu)
    print('Finished simulation')
    return msim, debug

def psc_europeanlike_220607(N,T,lambda_array,mutation_map,mu,p,simulation_model="hudson",verbosity=False,recordfullarg=False):
  
    demography = msprime.Demography()
    demography.add_population(name="A", initial_size=N)
    
    for i in range(0,len(lambda_array)):
        demography.add_population_parameters_change(T[i], initial_size=N*(1/lambda_array[i]), growth_rate=None, population='A')
    demography.sort_events()
    debug = demography.debug()
    if verbosity: print(debug)

    print('Starting simulation')
    sim = msprime.sim_ancestry(samples={"A": 1},ploidy=2,demography=demography,sequence_length=int(mutation_map.right[-1]),recombination_rate=p,record_full_arg=recordfullarg,model=simulation_model)
    # msim = msprime.sim_mutations(sim, rate=mutation_map)

    msim_mutmap = msprime.sim_mutations(sim, rate=mutation_map)
    msim_nomutmap = msprime.sim_mutations(sim, rate=mu)

    print('Finished simulation')
    return msim_mutmap, msim_nomutmap, debug

# def psc_europeanlike_220607_nomutationmap(N,T,lambda_array,L,mu,p,simulation_model="hudson",verbosity=False,recordfullarg=False):
  
#     demography = msprime.Demography()
#     demography.add_population(name="A", initial_size=N)
    
#     for i in range(0,len(lambda_array)):
#         demography.add_population_parameters_change(T[i], initial_size=N*(1/lambda_array[i]), growth_rate=None, population='A')
#     demography.sort_events()
#     debug = demography.debug()
#     if verbosity: print(debug)
#     print('Starting simulation')
#     sim = msprime.sim_ancestry(samples={"A": 1},ploidy=2,demography=demography,sequence_length=L,recombination_rate=p,record_full_arg=recordfullarg,model=simulation_model)
#     msim = msprime.sim_mutations(sim, rate=mu)
#     print('Finished simulation')
#     return msim, debug

def psc_bottleneck_220427(N,mu,T,lambda_array,recombination_map,simulation_model="hudson",verbosity=False,recordfullarg=False):
    
    demography = msprime.Demography()
    demography.add_population(name="A", initial_size=N)
    
    for i in range(0,len(lambda_array)):
        demography.add_population_parameters_change(T[i], initial_size=N*(1/lambda_array[i]), growth_rate=None, population='A')
    demography.sort_events()
    debug = demography.debug()
    if verbosity: print(debug)
    print('Starting simulation')
    sim = msprime.sim_ancestry(samples={"A": 1},ploidy=2,demography=demography,sequence_length=recombination_map.right[-1],recombination_rate=recombination_map,record_full_arg=recordfullarg,model=simulation_model)
    msim = msprime.sim_mutations(sim, rate=mu)
    print('Finished simulation')
    return msim, debug

def psc_expansion_220427(N,mu,T,lambda_array,recombination_map,simulation_model="hudson",verbosity=False,recordfullarg=False):
    
    demography = msprime.Demography()
    demography.add_population(name="A", initial_size=N)
    
    for i in range(0,len(lambda_array)):
        demography.add_population_parameters_change(T[i], initial_size=N*(1/lambda_array[i]), growth_rate=None, population='A')
    demography.sort_events()
    debug = demography.debug()
    if verbosity: print(debug)
    print('Starting simulation')
    sim = msprime.sim_ancestry(samples={"A": 1},ploidy=2,demography=demography,sequence_length=recombination_map.right[-1],recombination_rate=recombination_map,record_full_arg=recordfullarg,model=simulation_model)
    msim = msprime.sim_mutations(sim, rate=mu)
    print('Finished simulation')
    return msim, debug

def split_psc_220502(N,mu,gamma,T,lambda_array_A,lambda_array_B,recombination_map,simulation_model="hudson",verbosity=False,recordfullarg=False):
    
    start_index=10
    end_index=24
    demography = msprime.Demography()
    demography.add_population(name="A", initial_size=N)
    demography.add_population(name="B", initial_size=N)
    demography.add_mass_migration(time=T[start_index], source="A", dest="B", proportion=gamma)
    demography.add_mass_migration(time=T[end_index], source="B", dest="A", proportion=1)
    for i in range(0,len(lambda_array_A)):
        demography.add_population_parameters_change(T[i], initial_size=N*(1/lambda_array_A[i]), growth_rate=None, population='A')
    for i in range(0,len(lambda_array_B)):
        demography.add_population_parameters_change(T[i], initial_size=N*(1/lambda_array_B[i]), growth_rate=None, population='B')

    demography.sort_events()
    debug = demography.debug()
    if verbosity: print(debug)
    print('Starting simulation')
    sim = msprime.sim_ancestry(samples={"A": 1, "B": 0},ploidy=2,demography=demography,sequence_length=recombination_map.right[-1],recombination_rate=recombination_map,record_full_arg=recordfullarg,model=simulation_model)
    msim = msprime.sim_mutations(sim, rate=mu)
    print('Finished simulation')
    return msim, debug

def split_psc_220504(N,mu,gamma,recombination_map,T,simulation_model="hudson",verbosity=False,recordfullarg=False):
    
    start_index=10
    end_index=24
    demography = msprime.Demography()
    demography.add_population(name="A", initial_size=N)
    demography.add_population(name="B", initial_size=N)
    demography.add_mass_migration(time=T[start_index], source="A", dest="B", proportion=gamma)
    demography.add_mass_migration(time=T[end_index], source="B", dest="A", proportion=1)
    demography.sort_events()
    debug = demography.debug()
    if verbosity: print(debug)
    print('Starting simulation')
    sim = msprime.sim_ancestry(samples={"A": 1, "B": 0},ploidy=2,demography=demography,sequence_length=recombination_map.right[-1],recombination_rate=recombination_map,record_full_arg=recordfullarg,model=simulation_model)
    msim = msprime.sim_mutations(sim, rate=mu)
    print('Finished simulation')
    return msim, debug


def psc_220207(N,mu,p,L,T,lambda_array,simulation_model="hudson",verbosity=False):
    
    demography = msprime.Demography()
    demography.add_population(name="A", initial_size=N)
    
    for i in range(0,len(lambda_array)):
        demography.add_population_parameters_change(T[i], initial_size=N*(1/lambda_array[i]), growth_rate=None, population='A')
    demography.sort_events()
    debug = demography.debug()
    if verbosity: print(debug)
    print('Starting simulation')
    sim = msprime.sim_ancestry(samples={"A": 1},ploidy=2,demography=demography,sequence_length=L,recombination_rate=p,record_full_arg=True,model=simulation_model)
    msim = msprime.sim_mutations(sim, rate=mu)
    print('Finished simulation')
    return msim, debug

def write_mhs(pos,filename,chrom,ratemap=False,gzip_flag=True):
    # pos is index of hets
    # chrom is int
    current_chr = f'chr{chrom}'
    diff_pos = pos[1:] - pos[0:-1]
    SSPSS = np.concatenate(([pos[0]] ,diff_pos))
    gt = ['01']*len(pos)
    chr_label = [current_chr]*len(pos)
	

    if ratemap is not False:
        if np.isnan(ratemap.rate[0])==True:
            SSPSS[0] = pos[0] - ratemap.right[0]

    if gzip_flag==False:    
        with open(filename,'w') as f:
            lis=[chr_label,pos,SSPSS,gt]
            for x in zip(*lis):
                f.write("{0}\t{1}\t{2}\t{3}\n".format(*x))
    else:
        with gzip.open(filename,'wt') as f:
            lis=[chr_label,pos,SSPSS,gt]
            for x in zip(*lis):
                f.write("{0}\t{1}\t{2}\t{3}\n".format(*x))

    return None

def constpop_220519(N,mu,p,L,T,simulation_model="hudson",verbosity=True):

    demography = msprime.Demography()
    demography.add_population(name="A", initial_size=N)

    debug = demography.debug()
    if verbosity: print(debug)
    print(f'N={N};mu={mu};p={p};L={L};T={T}')
    print('Starting simulation')
    sim = msprime.sim_ancestry(samples={"A": 1},ploidy=2,demography=demography,sequence_length=L,recombination_rate=p,record_full_arg=False,model=simulation_model)
    msim = msprime.sim_mutations(sim, rate=mu)
    print('Finished simulation')
    return msim, debug
    # return sim, debug

def halvingpop_220519(N,mu,p,L,T,simulation_model="hudson",verbosity=True):

    D = len(T)
    D = len(T) - 1
    demography = msprime.Demography()
    demography.add_population(name="A", initial_size=2*N)
    demography.add_population_parameters_change(T[16], initial_size=N, growth_rate=None, population='A')

    debug = demography.debug()
    if verbosity: print(debug)
    print('Starting simulation')
    sim = msprime.sim_ancestry(samples={"A": 1},ploidy=2,demography=demography,sequence_length=L,recombination_rate=p,record_full_arg=False,model=simulation_model)
    msim = msprime.sim_mutations(sim, rate=mu)
    print('Finished simulation')
    return msim, debug
    # return sim, debug

def quarteringpop_220519(N,mu,p,L,T,simulation_model="hudson",verbosity=True):

    D = len(T)
    D = len(T) - 1
    demography = msprime.Demography()
    demography.add_population(name="A", initial_size=0.25*N)
    demography.add_population_parameters_change(T[16], initial_size=N, growth_rate=None, population='A')

    debug = demography.debug()
    if verbosity: print(debug)
    print(f'N={N};mu={mu};p={p};L={L};T={T}')
    print('Starting simulation')
    sim = msprime.sim_ancestry(samples={"A": 1},ploidy=2,demography=demography,sequence_length=L,recombination_rate=p,record_full_arg=False,model=simulation_model)
    msim = msprime.sim_mutations(sim, rate=mu)
    print('Finished simulation')
    return msim, debug
    # return sim, debug

def triplingpop_220519(N,mu,p,L,T,simulation_model="hudson",verbosity=True):

    D = len(T)
    D = len(T) - 1
    demography = msprime.Demography()
    demography.add_population(name="A", initial_size=3*N)
    demography.add_population_parameters_change(T[16], initial_size=N, growth_rate=None, population='A')

    debug = demography.debug()
    if verbosity: print(debug)
    print(f'N={N};mu={mu};p={p};L={L};T={T}')
    print('Starting simulation')
    sim = msprime.sim_ancestry(samples={"A": 1},ploidy=2,demography=demography,sequence_length=L,recombination_rate=p,record_full_arg=False,model=simulation_model)
    msim = msprime.sim_mutations(sim, rate=mu)
    print('Finished simulation')
    return msim, debug
    # return sim, debug

def triplingpop_221121(N,mu,p,L,T,simulation_model="hudson",verbosity=True):

    D = len(T)
    D = len(T) - 1
    demography = msprime.Demography()
    demography.add_population(name="A", initial_size=3*N)
    demography.add_population_parameters_change(T[13], initial_size=N, growth_rate=None, population='A')

    debug = demography.debug()
    if verbosity: print(debug)
    print(f'N={N};mu={mu};p={p};L={L};T={T}')
    print('Starting simulation')
    sim = msprime.sim_ancestry(samples={"A": 1},ploidy=2,demography=demography,sequence_length=L,recombination_rate=p,record_full_arg=False,model=simulation_model)
    msim = msprime.sim_mutations(sim, rate=mu)
    print('Finished simulation')
    return msim, debug
    # return sim, debug

def popchanges_221113(N,mu,p,L,T,simulation_model="hudson",verbosity=True):

    D = len(T)
    D = len(T) - 1
    demography = msprime.Demography()
    demography.add_population(name="A", initial_size=0.25*N)
    demography.add_population_parameters_change(T[13], initial_size=N, growth_rate=None, population='A')
    demography.add_population_parameters_change(T[20], initial_size=0.5*N, growth_rate=None, population='A')

    debug = demography.debug()
    if verbosity: print(debug)
    print(f'N={N};mu={mu};p={p};L={L};T={T}')
    print('Starting simulation')
    sim = msprime.sim_ancestry(samples={"A": 1},ploidy=2,demography=demography,sequence_length=L,recombination_rate=p,record_full_arg=False,model=simulation_model)
    msim = msprime.sim_mutations(sim, rate=mu)
    print('Finished simulation')
    return msim, debug
    # return sim, debug



def pop_simple_split_220712(N,pop_sizes,mu,p,L,T,start_index,end_index,gamma,simulation_model="hudson",return_debug=False,recordfullarg=False,verbosity=True):
    # T in gens
    # I use N as haploid here
    
    print(f'\t\tstart_index is {start_index}')
    print(f'\t\tend_index is {end_index}')
    print(f'\t\tgamma is {gamma}')
    print(f'\t\tsimulation model is {simulation_model}')


    demography = msprime.Demography()
    demography.add_population(name="A", initial_size=pop_sizes[0])
    demography.add_population(name="B", initial_size=pop_sizes[0])
    demography.add_mass_migration(time=T[start_index], source="A", dest="B", proportion=gamma)
    demography.add_mass_migration(time=T[end_index], source="B", dest="A", proportion=1)

    debug = demography.debug()
    if verbosity: print(debug)
    print('Starting simulation')
    sim = msprime.sim_ancestry(samples={"A": 1, "B": 0},ploidy=2,demography=demography,sequence_length=L,recombination_rate=p,record_full_arg=recordfullarg,model=simulation_model)
    msim = msprime.sim_mutations(sim, rate=mu)
    print('Finished simulation')
    # can return debug too for coalescence_rate_trajectory and stuff
    if return_debug is False:
        return msim
    else:
        return msim, debug

def pop_simpleconst_220712(N,pop_sizes,mu,p,L,T,simulation_model="hudson",verbosity=False,return_debug=False,recordfullarg=False):
    # T in gens
    # I use N as haploid here

    demography = msprime.Demography()
    demography.add_population(name="A", initial_size=pop_sizes[0])
    
    for i in range(0,len(pop_sizes)):
        demography.add_population_parameters_change(T[i], initial_size=pop_sizes[i], growth_rate=None, population='A')
    demography.sort_events()
    debug = demography.debug()
    if verbosity: print(debug)
    print('Starting simulation')
    sim = msprime.sim_ancestry(samples={"A": 1},ploidy=2,demography=demography,sequence_length=L,recombination_rate=p,record_full_arg=recordfullarg,model=simulation_model)
    msim = msprime.sim_mutations(sim, rate=mu)
    print('Finished simulation')
    return msim, debug


def pop_split_nonafrican_icr_220712(N,pop_sizes,mu,p,L,T,start_index,end_index,simulation_model="hudson",return_debug=False,recordfullarg=False,verbosity=True):
    # T in gens
    # I use N as haploid here

    print(f'pop_split_nonafrican_icr_220712, N is {N}')
    gamma = 0.25
    
    demography = msprime.Demography()
    demography.add_population(name="A", initial_size=pop_sizes[0])
    demography.add_population(name="B", initial_size=N)
    demography.add_mass_migration(time=T[start_index], source="A", dest="B", proportion=gamma)
    demography.add_mass_migration(time=T[end_index], source="B", dest="A", proportion=1)
    for i in range(0,len(pop_sizes)):
        demography.add_population_parameters_change(T[i], initial_size=pop_sizes[i], growth_rate=None, population='A')
    demography.sort_events()
    debug = demography.debug()
    if verbosity: print(debug)
    print('Starting simulation')
    sim = msprime.sim_ancestry(samples={"A": 1, "B": 0},ploidy=2,demography=demography,sequence_length=L,recombination_rate=p,record_full_arg=recordfullarg,model=simulation_model)
    msim = msprime.sim_mutations(sim, rate=mu)
    print('Finished simulation')
    # can return debug too for coalescence_rate_trajectory and stuff
    if return_debug is False:
        return msim
    else:
        return msim, debug

def pop_nonafrican_icr_220712_matching_psc(N,pop_sizes,mu,p,L,T,simulation_model="hudson",return_debug=False,recordfullarg=False,verbosity=True):
    # T in gens
    # I use N as haploid here
#     pop_sizes = pop_sizes/2
    demography = msprime.Demography()
    demography.add_population(name="A", initial_size=pop_sizes[0])
    for i in range(0,len(pop_sizes)):
        demography.add_population_parameters_change(T[i], initial_size=pop_sizes[i], growth_rate=None, population='A')
    demography.sort_events()
    debug = demography.debug()
    if verbosity: print(debug)
    print('Starting simulation')
    sim = msprime.sim_ancestry(samples={"A": 1},ploidy=2,demography=demography,sequence_length=L,recombination_rate=p,record_full_arg=recordfullarg,model=simulation_model)
    msim = msprime.sim_mutations(sim, rate=mu)
    print('Finished simulation')
    # can return debug too for coalescence_rate_trajectory and stuff
    if return_debug is False:
        return msim
    else:
        return msim, debug

def print_coalescent_rate(command_string,T_sim_gens):
    exec(f'sim, debug_cr = {command_string}')
    RAA, _ = debug_cr.coalescence_rate_trajectory(T_sim_gens, {"A": 2})
    print(f'coalescent rate is \n {1/RAA[0:-1]}')
    return None



def time_intervals(D,spread_1,spread_2): 
    T = [0]
    # for i in range(0,D-1): 
    #     T.append( spread_1*np.exp( (i/D)*np.log(1 + spread_2/spread_1) - 1))
    # T.append(T[-1]*10) # append stupidly large last tMRCA to represent infinity
    for i in range(0,D): 
        T.append( spread_1*np.exp( (i/D)*np.log(1 + spread_2/spread_1) - 1))
    # T.append(T[-1]*10) # append stupidly large last tMRCA to represent infinity

    T_np = np.array(T)
    return T_np

# parse args

parser = argparse.ArgumentParser(description="Set options for jump_size and number_files")
parser.add_argument('-p','--recomb_rate',help='per gen per bp recombination rate (scientific notation)',required=False,type=str,default=1e-08)
parser.add_argument('-mu','--mut_rate',help='per gen per bp mutation rate (scientific notation)',required=True,type=str)
parser.add_argument('-L','--seq_length',help='sequence length (scientific notation) for each iteration',required=False,type=str,default=1e+06)
parser.add_argument('-N','--diploid_N',help='Diploid Ne (scientific notation)',required=False,type=str,default=1e+04)
parser.add_argument('-D','--number_time_windows',help='The number of time windows to use in inference. Suggested value = 32',nargs='?',const=32,type=int,default=32)

parser.add_argument('-s','--start_index',help='Change in demographic scenario (pop split) starts at time T[start_index]',required=False,type=int)
parser.add_argument('-e','--end_index',help='Change in demographic scenario (pop split) ends at time T[end_index]',required=False,type=int)
parser.add_argument('-s_psc','--start_index_psc',help='Change in demographic scenario (psc) starts at time T[s_psc]',required=False,type=int)
parser.add_argument('-e_psc','--end_index_psc',help='Change in demographic scenario (psc) ends at time T[e_psc]',required=False,type=int)
parser.add_argument('-lambdas_A','--lambdas_A',help='string of lambda inputs for A, comma delimited',required=False,type=str)
parser.add_argument('-lambdas_B','--lambdas_B',help='string of lambda inputs for B, comma delimited',required=False,type=str)


parser.add_argument('-alpha','--intensity',help='Change in pop size parameter (or split fraction)',required=False,type=float)
parser.add_argument('-beta','--sub_intensity',help='Change in pop size parameter (or split fraction)',required=False,type=float)
parser.add_argument('-pop_change','--pop_change',help='Which population to change size, in split_psc model',required=False,type=str)

parser.add_argument('-spread1','--D_spread_1',help='Parameter controlling the time interval boundaries. Suggsted value = 0.1',nargs='?',const=0.1,type=float,default=0.1)
parser.add_argument('-spread2','--D_spread_2',help='Parameter controlling the time interval boundaries. Suggsted value = 20',nargs='?',const=20,type=float,default=20)
parser.add_argument('-recomb_map','--recomb_map',help='Use a recombination map',default=False,type=str) # '/home/tc557/ancestry/genetic_map_220426/genetic_map_hg38_chrom22.txt'
parser.add_argument('-mutation_map','--mutation_map',help='Use a mutation map',action='store_true',default=False) 
parser.add_argument('-mutation_map_alpha','--mutation_map_alpha',help='distribution of mutation map',default=5,type=float) 
parser.add_argument('-mutation_map_beta','--mutation_map_beta',help='distribution of mutation map',default=5,type=float) 
parser.add_argument('-mutation_map_change_rate','--mutation_map_change_rate',help='change mutation rate every ~Exp(mutation_map_change_rate) base pairs (give string of scientific notation)',default='1e+05',type=str) 
parser.add_argument('-mutation_map_num_mu_domains','--mutation_map_num_mu_domains',help='how many different possible mutation rates',default=100,type=int) 
parser.add_argument('-chrom','--chrom',help='When using a ratemap, which chromosome are you simulating',type=int,required=False,default=1)
parser.add_argument('-sample','--sample',help='Label for mhs files, if writing more than 1 it may be helpful',type=int,required=False,default=1)

parser.add_argument('-model','--model',help='Given model to simulate, must be string matching model in this script',type=str,required=True)
parser.add_argument('-sim_model','--simulation_model',help='msprime model to simulate from. Default is "hudson"',type=str,required=False,default="hudson")
parser.add_argument('-recordfullarg','--recordfullarg',help='Boolean as to whether record the full ARG or not.',default=False,action='store_true')

# parser.add_argument('-o','--output_path',help='output directory for hets and coal_data',required=False)
parser.add_argument('-o_mhs','--output_path_mhs',help='output path for mhs data',required=False)
parser.add_argument('-o_mhs_nomutmap','--output_path_mhs_nomutmap',help='output path for mhs data for nomutmap. Only needed if mutmap given',required=False)

parser.add_argument('-o_mhs_file','--output_file_mhs',help='output path for mhs data',required=False)

parser.add_argument('-o_hets','--output_path_hets',help='output path for all het data',required=False)
parser.add_argument('-o_hets1','--output_path_hets1',help='output path for het data1',required=False)
parser.add_argument('-o_hets2','--output_path_hets2',help='output path for het data2',required=False)
parser.add_argument('-o_hets3','--output_path_hets3',help='output path for het data3',required=False)
parser.add_argument('-lineage_path','--lineage_path',help='Record lineage path',default=False,action='store_true')


parser.add_argument('-o_tmrca','--output_path_tmrca',help='output path for tmrca data',required=False)
parser.add_argument('-w','--write_output',help='Flag for whether write files or not.',default=True,action='store_false')

parser.add_argument('-w_mhs','--write_mhs',help='Flag for whether write mhs files or not.',default=False,action='store_true')

parser.add_argument('-dont_save_het_data','--dont_save_het_data',help='Flag for whether save het files or (will still save tmrca_data).',default=False,action='store_true')

parser.add_argument('-its','--iterations',help='total number of iterations to do',nargs='?',type=int)

args = parser.parse_args()

zargs = dir(args)
zargs = [zarg for zarg in zargs if zarg[0]!='_']
for zarg in zargs:
    print(f'{zarg} is ',end='')
    # exec(f'{zarg}=args.{zarg}')
    exec(f'print(args.{zarg})')


p = float(args.recomb_rate)
mu = float(args.mut_rate)
L = int(float(args.seq_length))
N = int(float(args.diploid_N))
D = args.number_time_windows
spread_1 = args.D_spread_1
spread_2 = args.D_spread_2
sim_model = args.simulation_model
recordfullarg=args.recordfullarg
chrom=args.chrom
sample=args.sample

if args.lambdas_A is not None:
    lambdas_string_A = args.lambdas_A
    lambdas_list_A=lambdas_string_A.split(',')
    lambdas_list_A=[float(i) for i in lambdas_list_A]
    lambdas_A = np.array(lambdas_list_A)
    if len(lambdas_A)!=D:
        print(f'problem! len(lambdas)={len(lambdas_A)} which is not equal to D={D}. Aborting. ')
        sys.exit()

    lambdas_string_B = args.lambdas_B
    lambdas_list_B=lambdas_string_B.split(',')
    lambdas_list_B=[float(i) for i in lambdas_list_B]
    lambdas_B = np.array(lambdas_list_B)
    if len(lambdas_B)!=D:
        print(f'problem! len(lambdas)={len(lambdas_B)} which is not equal to D={D}. Aborting. ')
        sys.exit()

if args.recomb_map is not False:
    recomb_map = args.recomb_map
else:
    recomb_map=False

if args.mutation_map is not False:
    mutation_map = True
    mutation_map_alpha = args.mutation_map_alpha
    mutation_map_beta = args.mutation_map_beta
    mutation_map_change_rate = int(float(args.mutation_map_change_rate))
    mutation_map_num_mu_domains = args.mutation_map_num_mu_domains
    output_path_mhs_nomutmap = args.output_path_mhs_nomutmap
else:
    mutation_map = False

# python /home/tc557/ancestry/msprime_simulation_220216_combined.py -mutation_map -mutation_map_alpha 5 -mutation_map_beta 5 -mutation_map_change_rate 100000.0 -mutation_map_num_mu_domains 100 -p 1e-08 -mu 1.25e-08 -model psc_europeanlike_220607 -chrom 8 -sample 1 -N 15000 -o_mhs /home/tc557/rds/hpc-work/msprime_simulations/220607/psc_europeanlike_220607/ -its 1 -D 32 -sim_model hudson -w_mhs

if '/' in args.model:
    model = open(args.model).read()[0:-1]
else:
    model = args.model

if args.iterations is not None:
    if args.iterations<1:
        print('error! iterations must be bigger than 0. Currently it is {args.iterations}',flush=True)
    iterations = args.iterations
else:
    iterations = 1

total_L = int(L*iterations)

if args.start_index is not None:
    start_index = args.start_index
if args.end_index is not None:
    end_index = args.end_index

if args.start_index_psc is not None: 
    start_index_psc = args.start_index_psc
if args.end_index_psc is not None:
    end_index_psc = args.end_index_psc

if args.intensity is not None:
    intensity = args.intensity

if args.sub_intensity is not None: 
    sub_intensity = args.sub_intensity

if args.pop_change is not None:
    pop_change = args.pop_change

lineage_path = args.lineage_path
print(f'Record lineage_path is {lineage_path}',flush=True)

output_hets = args.output_path_hets
output_hets1 = args.output_path_hets1
output_hets2 = args.output_path_hets2
output_hets3 = args.output_path_hets3
output_path_mhs = args.output_path_mhs
output_file_mhs = args.output_file_mhs

output_tmrca = args.output_path_tmrca

write_output = args.write_output
dont_save_het_data = args.dont_save_het_data

print(f'Given the following input parameters:',flush=True)
print(f'\tp is {p}',flush=True)
print(f'\tmu is {mu}',flush=True)
print(f'\tL is {L}',flush=True)
print(f'\tN is {N}',flush=True)
print(f'\tD is {D}',flush=True)
print(f'\tmodel is {model}',flush=True)
print(f'\toutput paths are:',flush=True)
print(f'\t\tfor hets = {output_hets}',flush=True)
print(f'\t\tfor tmrca = {output_tmrca}',flush=True)
print(f'\titerations is {iterations}',flush=True)
print(f'\tTotal (combining all iterations) sequence length is {total_L}',flush=True)


seq_lengths_string = ["{:.0e}".format(total_L/(4-i)) for i in range(1,4)]
seq_lengths_int = [int(total_L/(4-i)) for i in range(1,4)]

if output_hets is None:
    output_hets = os.getcwd() + f'/hets_all.txt'
if output_hets1 is None:
    output_hets1 = os.getcwd() + f'/hets_SL{seq_lengths_string[0]}' + '.txt'
if output_hets2 is None:
    output_hets2 = os.getcwd() + f'/hets_SL{seq_lengths_string[1]}' + '.txt'
if output_hets3 is None:
    output_hets3 = os.getcwd() + f'/hets_SL{seq_lengths_string[2]}' + '.txt'
if output_tmrca is None:
    output_tmrca = os.getcwd() + f'/iteration_{iterations}' + '_tmrca.txt'


verbosity_present = True
if model=='const_pop220110':
    sim =  const_pop220110(p=p,mu=mu,L=L,N=N)
elif model=='pop_expansion220110':
    tm_true = Transition_Matrix_211220(D,spread_1,spread_2) # initialise transition matrix object
    T = tm_true.T # time array
    # print(f'\tT is {T}')
    sim = pop_expansion220110(N=N,mu=mu,p=p,L=L,T=T*2*N,start_index=start_index,end_index=end_index,intensity=intensity,simulation_model=sim_model)
elif model=='pop_split_220113':
    tm_true = Transition_Matrix_211220(D,spread_1,spread_2) # initialise transition matrix object
    T = tm_true.T # time array
    sim = pop_split_220113(N_A=N,N_B=N,mu=mu,p=p,L=L,T=T*2*N,start_index=start_index,end_index=end_index,gamma=intensity,simulation_model=sim_model)
elif model=='pop_split_220114':
    tm_true = Transition_Matrix_211220(D,spread_1,spread_2) # initialise transition matrix object
    T = tm_true.T # time array
    sim, debug = pop_split_220114(N_A=N,N_B=N,mu=mu,p=p,L=L,T=T*2*N,start_index=start_index,end_index=end_index,gamma=intensity,simulation_model=sim_model)
elif model=='pop_split_psc_pop_220119':
    tm_true = Transition_Matrix_211220(D,spread_1,spread_2) # initialise transition matrix object
    T = tm_true.T # time array
    print(f'T_coalescent = {T}')
    print(f'T_gens = T_coalescent*2*N = {T*2*N}')
    print(f'Admixture time (gens) = T_gens[{start_index}] = {T[int(start_index)]*2*N}')
    print(f'Divergence time (gens) = T_gens[{end_index}] = {T[int(end_index)]*2*N}')
    sim, debug = pop_split_psc_pop_220119(N=N,mu=mu,p=p,L=L,T=T*2*N,start_index_str=start_index,end_index_str=end_index, \
    start_index_psc=start_index_psc,end_index_psc=end_index_psc,gamma=intensity,intensity=sub_intensity, \
    pop_to_change=pop_change,simulation_model=sim_model,verbosity=False)
    command_string=''
elif model=='matching_psc_pop_split_220120':
    tm_true = Transition_Matrix_211220(D,spread_1,spread_2) # initialise transition matrix object
    T = tm_true.T # time array    
    sim_split,debug_split = pop_split_220114(N_A=N,N_B=N,mu=mu,p=p,L=1,T=T*2*N,start_index=start_index,end_index=end_index,gamma=intensity,return_debug=True)
    # T = np.linspace(0, 50000, 51)
    RAA_split, _ = debug_split.coalescence_rate_trajectory(T*2*N, {"A": 2})
    sim,debug = matching_psc_pop_split_220120(N_A=N,N_B=N,mu=mu,p=p,L=L,T=T*2*N,start_index=start_index,end_index=end_index,gamma=intensity,pop_size_changes=(1/(2*RAA_split)),simulation_model="hudson")
    # del RAA_split
    RAA, _ = debug.coalescence_rate_trajectory(T*2*N, {"A": 2})
elif model=='pop_split_psc_pop_220207':
    # this calls the structured model but then actually generates from the panmictic one
    tm_true = Transition_Matrix_211220(D,spread_1,spread_2) # initialise transition matrix object
    T = tm_true.T # time array
    # got N, got L, got mu, got p, got ts_index, got te_index, got intensity (called "sub-intensity"), got gamma (called "intensity")

    pop_sizes = np.ones(D)
    pop_sizes[0:3] = 0.8
    pop_sizes[3:start_index] = sub_intensity
    lambda_A_true = 1/pop_sizes
    pop_to_change = 'A'
    sim_split,debug_split = pop_split_psc_pop_220207(N=N,mu=mu,p=p,L=1,T=T*2*N,start_index_str=start_index,end_index_str=end_index, \
                         gamma=intensity,lambda_array=lambda_A_true, \
                         pop_to_change=pop_to_change,simulation_model="hudson",verbosity=True)
    RAA_split, _ = debug_split.coalescence_rate_trajectory(T*2*N, {"A": 2})
    lambda_A_psc = (2*N*RAA_split)
    sim, debug = psc_220207(N=N,mu=mu,p=p,L=1,T=T*2*N,lambda_array=lambda_A_psc,simulation_model="hudson",verbosity=True)
    RAA, _ = debug.coalescence_rate_trajectory(T*2*N, {"A": 2})
    print_iCR=True
    if print_iCR:
        print(f'1/(2*RAA_split) =\n{1/(2*RAA_split)}')
        print(f'1/(2*RAA_psc) =\n{1/(2*RAA)}')
    command_string_structure = f'pop_split_psc_pop_220207(N={N},mu={mu},p={p},L={L},T={T*2*N},start_index_str={start_index},end_index_str={end_index},gamma={intensity},lambda_array={lambda_A_true},pop_to_change={pop_to_change},simulation_model="hudson",verbosity=False)'
    command_string = f'psc_220207(N=N,mu=mu,p=p,L=L,T=T*2*N,lambda_array=lambda_A_psc,simulation_model="hudson",verbosity=False)'
elif model=='pop_split_psc_pop_220207_true':
    # this actually simulates from the structured model
    tm_true = Transition_Matrix_211220(D,spread_1,spread_2) # initialise transition matrix object
    T = tm_true.T # time array
    # got N, got L, got mu, got p, got ts_index, got te_index, got intensity (called "sub-intensity"), got gamma (called "intensity")

    pop_sizes = np.ones(D)
    pop_sizes[0:3] = 0.8
    pop_sizes[3:start_index] = sub_intensity
    lambda_A_true = 1/pop_sizes
    pop_to_change = 'A'
    sim_split,debug_split = pop_split_psc_pop_220207(N=N,mu=mu,p=p,L=1,T=T*2*N,start_index_str=start_index,end_index_str=end_index, \
                         gamma=intensity,lambda_array=lambda_A_true, \
                         pop_to_change=pop_to_change,simulation_model="hudson",verbosity=True)
    RAA_split, _ = debug_split.coalescence_rate_trajectory(T*2*N, {"A": 2})
    # lambda_A_psc = (2*N*RAA_split)
    # sim, debug = psc_220207(N=N,mu=mu,p=p,L=1,T=T*2*N,lambda_array=lambda_A_psc,simulation_model="hudson",verbosity=True)
    # RAA, _ = debug.coalescence_rate_trajectory(T*2*N, {"A": 2})
    print_iCR=True
    if print_iCR:
        print(f'1/(2*RAA_split) =\n{1/(2*RAA_split)}')
        # print(f'1/(2*RAA_psc) =\n{1/(2*RAA)}')
    command_string = f'pop_split_psc_pop_220207(N=N,mu=mu,p=p,L=L,T=T*2*N,start_index_str=start_index,end_index_str=end_index,gamma=intensity,lambda_array=lambda_A_true,pop_to_change=pop_to_change,simulation_model="hudson",verbosity=False)'
    # command_string = f'psc_220207(N=N,mu=mu,p=p,L=L,T=T*2*N,lambda_array=lambda_A_psc,simulation_model="hudson",verbosity=False)'
elif model=='pop_split_psc_pop_240604':
    # this calls the structured model but then actually generates from the panmictic one
    tm_true = Transition_Matrix_211220(D,spread_1,spread_2) # initialise transition matrix object

    T = tm_true.T # time array
    # got N, got L, got mu, got p, got ts_index, got te_index, got intensity (called "sub-intensity"), got gamma (called "intensity")
    print(f'T_coalescent = {T}')
    print(f'T_gens = T_coalescent*2*N = {T*2*N}')
    print(f'Admixture time (gens) = T_gens[{start_index}] = {T[int(start_index)]*2*N}')
    print(f'Divergence time (gens) = T_gens[{end_index}] = {T[int(end_index)]*2*N}')

    pop_sizes = np.ones(D)
    pop_sizes[0:3] = 2
    pop_sizes[3:start_index] = sub_intensity
    lambda_A_true = 1/pop_sizes
    pop_to_change = 'A'
    sim_split,debug_split = pop_split_psc_pop_220207(N=N,mu=mu,p=p,L=1,T=T*2*N,start_index_str=start_index,end_index_str=end_index, \
                         gamma=intensity,lambda_array=lambda_A_true, \
                         pop_to_change=pop_to_change,simulation_model="hudson",verbosity=False)
    RAA_split, _ = debug_split.coalescence_rate_trajectory(T*2*N, {"A": 2})
    lambda_A_psc = (2*N*RAA_split)
    sim, debug = psc_220207(N=N,mu=mu,p=p,L=1,T=T*2*N,lambda_array=lambda_A_psc,simulation_model="hudson",verbosity=False)
    RAA, _ = debug.coalescence_rate_trajectory(T*2*N, {"A": 2})
    print_iCR=True
    if print_iCR:
        print(f'1/(2*RAA_split) =\n{1/(2*RAA_split)}')
        print(f'1/(2*RAA_psc) =\n{1/(2*RAA)}')
    command_string_structure = f'pop_split_psc_pop_220207(N={N},mu={mu},p={p},L={L},T={T*2*N},start_index_str={start_index},end_index_str={end_index},gamma={intensity},lambda_array={lambda_A_true},pop_to_change={pop_to_change},simulation_model="hudson",verbosity=False)'
    command_string = f'psc_220207(N=N,mu=mu,p=p,L=L,T=T*2*N,lambda_array=lambda_A_psc,simulation_model="hudson",verbosity=False)'
elif model=='pop_split_psc_pop_240604_true':
    # this actually simulates from the structured model
    tm_true = Transition_Matrix_211220(D,spread_1,spread_2) # initialise transition matrix object
    T = tm_true.T # time array 
    # got N, got L, got mu, got p, got ts_index, got te_index, got intensity (called "sub-intensity"), got gamma (called "intensity")

    print(f'T_coalescent = {T}')
    print(f'T_gens = T_coalescent*2*N = {T*2*N}')
    print(f'Admixture time (gens) = T_gens[{start_index}] = {T[int(start_index)]*2*N}')
    print(f'Divergence time (gens) = T_gens[{end_index}] = {T[int(end_index)]*2*N}')

    pop_sizes = np.ones(D)
    pop_sizes[0:3] = 2
    pop_sizes[3:start_index] = sub_intensity
    lambda_A_true = 1/pop_sizes
    pop_to_change = 'A'
    sim_split,debug_split = pop_split_psc_pop_220207(N=N,mu=mu,p=p,L=1,T=T*2*N,start_index_str=start_index,end_index_str=end_index, \
                         gamma=intensity,lambda_array=lambda_A_true, \
                         pop_to_change=pop_to_change,simulation_model="hudson",verbosity=False)
    RAA_split, _ = debug_split.coalescence_rate_trajectory(T*2*N, {"A": 2})
    # lambda_A_psc = (2*N*RAA_split)
    # sim, debug = psc_220207(N=N,mu=mu,p=p,L=1,T=T*2*N,lambda_array=lambda_A_psc,simulation_model="hudson",verbosity=True)
    # RAA, _ = debug.coalescence_rate_trajectory(T*2*N, {"A": 2})
    print_iCR=True
    if print_iCR:
        print(f'1/(2*RAA_split) =\n{1/(2*RAA_split)}')
        # print(f'1/(2*RAA_psc) =\n{1/(2*RAA)}')
    command_string = f'pop_split_psc_pop_220207(N=N,mu=mu,p=p,L=L,T=T*2*N,start_index_str=start_index,end_index_str=end_index,gamma=intensity,lambda_array=lambda_A_true,pop_to_change=pop_to_change,simulation_model="hudson",verbosity=False)'
    # command_string = f'psc_220207(N=N,mu=mu,p=p,L=L,T=T*2*N,lambda_array=lambda_A_psc,simulation_model="hudson",verbosity=False)'
elif model=='pop_split_psc_pop_220312':
    tm_true = Transition_Matrix_211220(D,spread_1,spread_2) # initialise transition matrix object
    T = tm_true.T # time array
    print(f'pop_to_change is {pop_change}',flush=True)
    print(f'T_s_index is {start_index}',flush=True)
    print(f'T_e_index is {end_index}',flush=True)
    print(f'psc_start_index is {start_index_psc}',flush=True)
    print(f'psc_end_index is {end_index_psc}',flush=True)
    print(f'gamma is {intensity}',flush=True)
    print(f'intensity is {sub_intensity}',flush=True)
    print(f'L is {L}')
    print(f'sim_model is {sim_model}',flush=True)


    
    sim,debug = pop_split_psc_pop_220312(N=N,mu=mu,p=p,L=1,T=T*2*N,start_index_str=start_index,end_index_str=end_index, \
                            start_index_psc=start_index_psc,end_index_psc=end_index_psc,gamma=intensity,intensity=sub_intensity, \
                            pop_to_change=pop_change,simulation_model=sim_model)
    command_string = f'pop_split_psc_pop_220312(N=N,mu=mu,p=p,L=L,T=T*2*N,start_index_str=start_index,end_index_str=end_index, \
                            start_index_psc=start_index_psc,end_index_psc=end_index_psc,gamma=intensity,intensity=sub_intensity, \
                            pop_to_change=pop_change,simulation_model=sim_model,verbosity=False)'  
    RAA, _ = debug.coalescence_rate_trajectory(T*2*N, {"A": 2})
    # lambda_A_psc = (2*N*RAA_split)
    # sim, debug = psc_220207(N=N,mu=mu,p=p,L=1,T=T*2*N,lambda_array=lambda_A_psc,simulation_model="hudson",verbosity=True)
    # RAA, _ = debug.coalescence_rate_trajectory(T*2*N, {"A": 2})
    print_iCR=True
    if print_iCR:
        print(f'1/(2*RAA) =\n{1/(2*RAA)}',flush=True)
        # print(f'1/(2*RAA_psc) =\n{1/(2*RAA)}')
elif model=='pop_split_psc_pop_220425':
    tm_true = Transition_Matrix_211220(D,spread_1,spread_2) # initialise transition matrix object
    T = tm_true.T # time array
    print(f'pop_to_change is {pop_change}',flush=True)
    print(f'T_s_index is {start_index}',flush=True)
    print(f'T_e_index is {end_index}',flush=True)
    print(f'psc_start_index is {start_index_psc}',flush=True)
    print(f'psc_end_index is {end_index_psc}',flush=True)
    print(f'gamma is {intensity}',flush=True)
    print(f'intensity is {sub_intensity}',flush=True)
    print(f'L is {L}')
    print(f'sim_model is {sim_model}',flush=True)


    
    sim,debug = pop_split_psc_pop_220425(N=N,mu=mu,p=p,L=1,T=T*2*N,start_index_str=start_index,end_index_str=end_index, \
                            start_index_psc=start_index_psc,end_index_psc=end_index_psc,gamma=intensity,intensity=sub_intensity, \
                            pop_to_change=pop_change,simulation_model=sim_model)
    command_string = f'pop_split_psc_pop_220425(N=N,mu=mu,p=p,L=L,T=T*2*N,start_index_str=start_index,end_index_str=end_index, \
                            start_index_psc=start_index_psc,end_index_psc=end_index_psc,gamma=intensity,intensity=sub_intensity, \
                            pop_to_change=pop_change,simulation_model=sim_model,verbosity=False)'  
    RAA, _ = debug.coalescence_rate_trajectory(T*2*N, {"A": 2})
    # lambda_A_psc = (2*N*RAA_split)
    # sim, debug = psc_220207(N=N,mu=mu,p=p,L=1,T=T*2*N,lambda_array=lambda_A_psc,simulation_model="hudson",verbosity=True)
    # RAA, _ = debug.coalescence_rate_trajectory(T*2*N, {"A": 2})
    print_iCR=True
    if print_iCR:
        print(f'1/(2*RAA) =\n{1/(2*RAA)}',flush=True)
        # print(f'1/(2*RAA_psc) =\n{1/(2*RAA)}')


        
elif model=='pop_split_psc_pop_220612':
    tm_true = Transition_Matrix_211220(D,spread_1,spread_2) # initialise transition matrix object
    T = tm_true.T # time array
    print(f'pop_to_change is {pop_change}',flush=True)
    print(f'T_s_index is {start_index}',flush=True)
    print(f'T_e_index is {end_index}',flush=True)
    print(f'psc_start_index is {start_index_psc}',flush=True)
    print(f'psc_end_index is {end_index_psc}',flush=True)
    print(f'gamma is {intensity}',flush=True)
    print(f'intensity is {sub_intensity}',flush=True)
    print(f'L is {L}')
    print(f'sim_model is {sim_model}',flush=True)

    sim,debug = pop_split_psc_pop_220612(N=N,mu=mu,p=p,L=1,T=T*2*N,start_index_str=start_index,end_index_str=end_index, \
                            start_index_psc=start_index_psc,end_index_psc=end_index_psc,gamma=intensity,intensity=sub_intensity, \
                            pop_to_change=pop_change,simulation_model=sim_model)
    command_string = f'pop_split_psc_pop_220612(N=N,mu=mu,p=p,L=L,T=T*2*N,start_index_str=start_index,end_index_str=end_index, \
                            start_index_psc=start_index_psc,end_index_psc=end_index_psc,gamma=intensity,intensity=sub_intensity, \
                            pop_to_change=pop_change,simulation_model=sim_model,verbosity=False)'  
    RAA, _ = debug.coalescence_rate_trajectory(T*2*N, {"A": 2})
    # lambda_A_psc = (2*N*RAA_split)
    # sim, debug = psc_220207(N=N,mu=mu,p=p,L=1,T=T*2*N,lambda_array=lambda_A_psc,simulation_model="hudson",verbosity=True)
    # RAA, _ = debug.coalescence_rate_trajectory(T*2*N, {"A": 2})
    print_iCR=True
    if print_iCR:
        print(f'1/(2*RAA) =\n{1/(2*RAA)}',flush=True)
        # print(f'1/(2*RAA_psc) =\n{1/(2*RAA)}')

elif model=='pop_split_psc_pop_220613':
    tm_true = Transition_Matrix_211220(D,spread_1,spread_2) # initialise transition matrix object
    T = tm_true.T # time array
    print(f'pop_to_change is {pop_change}',flush=True)
    print(f'T_s_index is {start_index}',flush=True)
    print(f'T_e_index is {end_index}',flush=True)
    print(f'psc_start_index is {start_index_psc}',flush=True)
    print(f'psc_end_index is {end_index_psc}',flush=True)
    print(f'gamma is {intensity}',flush=True)
    print(f'intensity is {sub_intensity}',flush=True)
    print(f'L is {L}')
    print(f'sim_model is {sim_model}',flush=True)

    sim,debug = pop_split_psc_pop_220613(N=N,mu=mu,p=p,L=1,T=T*2*N,start_index_str=start_index,end_index_str=end_index, \
                            start_index_psc=start_index_psc,end_index_psc=end_index_psc,gamma=intensity,intensity=sub_intensity, \
                            pop_to_change=pop_change,simulation_model=sim_model)
    command_string = f'pop_split_psc_pop_220613(N=N,mu=mu,p=p,L=L,T=T*2*N,start_index_str=start_index,end_index_str=end_index, \
                            start_index_psc=start_index_psc,end_index_psc=end_index_psc,gamma=intensity,intensity=sub_intensity, \
                            pop_to_change=pop_change,simulation_model=sim_model,verbosity=False)'  
    RAA, _ = debug.coalescence_rate_trajectory(T*2*N, {"A": 2})
    # lambda_A_psc = (2*N*RAA_split)
    # sim, debug = psc_220207(N=N,mu=mu,p=p,L=1,T=T*2*N,lambda_array=lambda_A_psc,simulation_model="hudson",verbosity=True)
    # RAA, _ = debug.coalescence_rate_trajectory(T*2*N, {"A": 2})
    print_iCR=True
    if print_iCR:
        print(f'1/(2*RAA) =\n{1/(2*RAA)}',flush=True)
        # print(f'1/(2*RAA_psc) =\n{1/(2*RAA)}')

elif model=='pop_split_psc_pop_220614':
    tm_true = Transition_Matrix_211220(D,spread_1,spread_2) # initialise transition matrix object
    T = tm_true.T # time array
    print(f'pop_to_change is {pop_change}',flush=True)
    print(f'T_s_index is {start_index}',flush=True)
    print(f'T_e_index is {end_index}',flush=True)
    print(f'psc_start_index is {start_index_psc}',flush=True)
    print(f'psc_end_index is {end_index_psc}',flush=True)
    print(f'gamma is {intensity}',flush=True)
    print(f'intensity is {sub_intensity}',flush=True)
    print(f'L is {L}')
    print(f'sim_model is {sim_model}',flush=True)
    
    sim,debug = pop_split_psc_pop_220614(N=N,mu=mu,p=p,L=1,T=T*2*N,start_index_str=start_index,end_index_str=end_index, \
                            start_index_psc=start_index_psc,end_index_psc=end_index_psc,gamma=intensity,intensity=sub_intensity, \
                            pop_to_change=pop_change,simulation_model=sim_model)
    command_string = f'pop_split_psc_pop_220614(N=N,mu=mu,p=p,L=L,T=T*2*N,start_index_str=start_index,end_index_str=end_index, \
                            start_index_psc=start_index_psc,end_index_psc=end_index_psc,gamma=intensity,intensity=sub_intensity, \
                            pop_to_change=pop_change,simulation_model=sim_model,verbosity=False)'  
    RAA, _ = debug.coalescence_rate_trajectory(T*2*N, {"A": 2})
    # lambda_A_psc = (2*N*RAA_split)
    # sim, debug = psc_220207(N=N,mu=mu,p=p,L=1,T=T*2*N,lambda_array=lambda_A_psc,simulation_model="hudson",verbosity=True)
    # RAA, _ = debug.coalescence_rate_trajectory(T*2*N, {"A": 2})
    print_iCR=True
    if print_iCR:
        print(f'1/(2*RAA) =\n{1/(2*RAA)}',flush=True)
        # print(f'1/(2*RAA_psc) =\n{1/(2*RAA)}')

elif model=='pop_split_psc_pop_b220614':
    tm_true = Transition_Matrix_211220(D,spread_1,spread_2) # initialise transition matrix object
    T = tm_true.T # time array
    print(f'pop_to_change is {pop_change}',flush=True)
    print(f'T_s_index is {start_index}',flush=True)
    print(f'T_e_index is {end_index}',flush=True)
    print(f'psc_start_index is {start_index_psc}',flush=True)
    print(f'psc_end_index is {end_index_psc}',flush=True)
    print(f'gamma is {intensity}',flush=True)
    print(f'intensity is {sub_intensity}',flush=True)
    print(f'L is {L}')
    print(f'sim_model is {sim_model}',flush=True)
    
    sim,debug = pop_split_psc_pop_b220614(N=N,mu=mu,p=p,L=1,T=T*2*N,start_index_str=start_index,end_index_str=end_index, \
                            start_index_psc=start_index_psc,end_index_psc=end_index_psc,gamma=intensity,intensity=sub_intensity, \
                            pop_to_change=pop_change,simulation_model=sim_model)
    command_string = f'pop_split_psc_pop_b220614(N=N,mu=mu,p=p,L=L,T=T*2*N,start_index_str=start_index,end_index_str=end_index, \
                            start_index_psc=start_index_psc,end_index_psc=end_index_psc,gamma=intensity,intensity=sub_intensity, \
                            pop_to_change=pop_change,simulation_model=sim_model,verbosity=False)'  
    RAA, _ = debug.coalescence_rate_trajectory(T*2*N, {"A": 2})
    # lambda_A_psc = (2*N*RAA_split)
    # sim, debug = psc_220207(N=N,mu=mu,p=p,L=1,T=T*2*N,lambda_array=lambda_A_psc,simulation_model="hudson",verbosity=True)
    # RAA, _ = debug.coalescence_rate_trajectory(T*2*N, {"A": 2})
    print_iCR=True
    if print_iCR:
        print(f'1/(2*RAA) =\n{1/(2*RAA)}',flush=True)
        # print(f'1/(2*RAA_psc) =\n{1/(2*RAA)}')


elif model=='pop_custom_220429a':
    tm_true = Transition_Matrix_211220(D,spread_1,spread_2) # initialise transition matrix object
    T = tm_true.T # time array
    print(f'psc_start_index is {start_index_psc}',flush=True)
    print(f'psc_end_index is {end_index_psc}',flush=True)
    print(f'intensity is {sub_intensity}',flush=True)
    print(f'L is {L}')
    print(f'sim_model is {sim_model}',flush=True)

    # pop_custom_220429a(N,mu,p,L,T,intensity,simulation_model="hudson",verbosity=True):
    
    sim,debug = pop_custom_220429a(N=N,mu=mu,p=p,L=1,intensity=sub_intensity,T=T*2*N,simulation_model=sim_model)
    command_string = f'pop_custom_220429a(N=N,mu=mu,p=p,L=L,intensity=sub_intensity,T=T*2*N,simulation_model=sim_model,verbosity=False)'  
    RAA, _ = debug.coalescence_rate_trajectory(T*2*N, {"A": 2})
    # lambda_A_psc = (2*N*RAA_split)
    # sim, debug = psc_220207(N=N,mu=mu,p=p,L=1,T=T*2*N,lambda_array=lambda_A_psc,simulation_model="hudson",verbosity=True)
    # RAA, _ = debug.coalescence_rate_trajectory(T*2*N, {"A": 2})
    print_iCR=True
    if print_iCR:
        print(f'1/(2*RAA) =\n{1/(2*RAA)}',flush=True)
        # print(f'1/(2*RAA_psc) =\n{1/(2*RAA)}')
elif model=='pop_custom_220429b':
    tm_true = Transition_Matrix_211220(D,spread_1,spread_2) # initialise transition matrix object
    T = tm_true.T # time array
    print(f'psc_start_index is {start_index_psc}',flush=True)
    print(f'psc_end_index is {end_index_psc}',flush=True)
    print(f'intensity is {sub_intensity}',flush=True)
    print(f'L is {L}')
    print(f'sim_model is {sim_model}',flush=True)
    
    sim,debug = pop_custom_220429b(N=N,mu=mu,p=p,intensity=sub_intensity,L=1,T=T*2*N,simulation_model=sim_model)
    command_string = f'pop_custom_220429b(N=N,mu=mu,p=p,L=L,intensity=sub_intensity,T=T*2*N,simulation_model=sim_model,verbosity=False)'  
    RAA, _ = debug.coalescence_rate_trajectory(T*2*N, {"A": 2})
    # lambda_A_psc = (2*N*RAA_split)
    # sim, debug = psc_220207(N=N,mu=mu,p=p,L=1,T=T*2*N,lambda_array=lambda_A_psc,simulation_model="hudson",verbosity=True)
    # RAA, _ = debug.coalescence_rate_trajectory(T*2*N, {"A": 2})
    print_iCR=True
    if print_iCR:
        print(f'1/(2*RAA) =\n{1/(2*RAA)}',flush=True)
        # print(f'1/(2*RAA_psc) =\n{1/(2*RAA)}')


elif model=='psc_europeanlike_220427':
    tm_true = Transition_Matrix_211220(D,spread_1,spread_2) # initialise transition matrix object
    T = tm_true.T # time array

    lambda_array = np.ones(D)
    lambda_array[0:6] = 0.3
    lambda_array[6:12] = 5
    lambda_array[12:24] = 0.7
    print(f'lambda_array is {lambda_array}',flush=True)
    print(f'i.e. pop sizes are {(1/lambda_array)*N}',flush=True)
    print(f'N is {N}',flush=True)
    print(f'mu is {mu}',flush=True)

    if args.recomb_map is False:
        print(f'Error. Recomb map needed for this model={model}',flush=True)
        sys.exit()
    print(f'using chromosome {chrom}',flush=True)
    print(f'using recombination map: {recomb_map}',flush=True)
    rate_map = msprime.RateMap.read_hapmap(recomb_map)
    # sliced_rate_map = rate_map.slice(0,1e+05)
    # sim,debug = psc_europeanlike_220427(N=N,mu=mu,T=T,recombination_map=sliced_rate_map,simulation_model="hudson", \
                        # verbosity=False,recordfullarg=recordfullarg)
    # RAA, _ = debug.coalescence_rate_trajectory(T*2*N, {"A": 2})
    # print_iCR=True
    # if print_iCR:
        # print(f'1/(2*RAA) =\n{1/(2*RAA)}',flush=True)
    command_string = f'psc_europeanlike_220427(N=N,lambda_array=lambda_array,mu=mu,T=T*2*N,recombination_map=rate_map,simulation_model="hudson", \
                        verbosity=True,recordfullarg=recordfullarg)'
elif model=='psc_bottleneck_220427':
    tm_true = Transition_Matrix_211220(D,spread_1,spread_2) # initialise transition matrix object
    T = tm_true.T # time array

    lambda_array = np.ones(D)
    lambda_array[10:20] = 8
    print(f'lambda_array is {lambda_array}',flush=True)
    print(f'i.e. pop sizes are {(1/lambda_array)*N}',flush=True)
    print(f'N is {N}',flush=True)
    print(f'mu is {mu}',flush=True)

    if args.recomb_map is False:
        print(f'Error. Recomb map needed for this model={model}',flush=True)
        sys.exit()
    print(f'using chromosome {chrom}',flush=True)
    print(f'using recombination map: {recomb_map}',flush=True)
    rate_map = msprime.RateMap.read_hapmap(recomb_map)
    # sliced_rate_map = rate_map.slice(0,1e+05)
    # sim,debug = psc_europeanlike_220427(N=N,mu=mu,T=T,recombination_map=sliced_rate_map,simulation_model="hudson", \
                        # verbosity=False,recordfullarg=recordfullarg)
    # RAA, _ = debug.coalescence_rate_trajectory(T*2*N, {"A": 2})
    # print_iCR=True
    # if print_iCR:
        # print(f'1/(2*RAA) =\n{1/(2*RAA)}',flush=True)
    command_string = f'psc_bottleneck_220427(N=N,mu=mu,lambda_array=lambda_array,T=T*2*N,recombination_map=rate_map,simulation_model="hudson", \
                        verbosity=True,recordfullarg=recordfullarg)'
elif model=='psc_expansion_220427':
    tm_true = Transition_Matrix_211220(D,spread_1,spread_2) # initialise transition matrix object
    T = tm_true.T # time array

    lambda_array = np.ones(D)
    lambda_array[10:20] = 1/2.5
    print(f'lambda_array is {lambda_array}',flush=True)
    print(f'i.e. pop sizes are {(1/lambda_array)*N}',flush=True)
    print(f'N is {N}',flush=True)
    print(f'mu is {mu}',flush=True)

    if args.recomb_map is False:
        print(f'Error. Recomb map needed for this model={model}',flush=True)
        sys.exit()
    print(f'using chromosome {chrom}',flush=True)
    print(f'using recombination map: {recomb_map}',flush=True)
    rate_map = msprime.RateMap.read_hapmap(recomb_map)
    # sliced_rate_map = rate_map.slice(0,1e+05)
    # sim,debug = psc_europeanlike_220427(N=N,mu=mu,T=T,recombination_map=sliced_rate_map,simulation_model="hudson", \
                        # verbosity=False,recordfullarg=recordfullarg)
    # RAA, _ = debug.coalescence_rate_trajectory(T*2*N, {"A": 2})
    # print_iCR=True
    # if print_iCR:
        # print(f'1/(2*RAA) =\n{1/(2*RAA)}',flush=True)
    command_string = f'psc_expansion_220427(N=N,mu=mu,lambda_array=lambda_array,T=T*2*N,recombination_map=rate_map,simulation_model="hudson", \
                        verbosity=True,recordfullarg=recordfullarg)'
elif model=='split_psc_Achange_220502':
    tm_true = Transition_Matrix_211220(D,spread_1,spread_2) # initialise transition matrix object
    T = tm_true.T # time array
    ts = 10
    te = 24
    A_popsize = np.ones(D)
    B_popsize = np.ones(D)

    A_popsize[0:4] = 3
    A_popsize[4:10] = 0.2
    A_popsize[10:13] = 1
    A_popsize[13:20] = 2
    A_popsize[te:te+3] = 1.5
    lambda_array_A = 1/A_popsize
    lambda_array_B = 1/B_popsize

    print(f'lambda_array_A is {lambda_array_A}',flush=True)
    print(f'lambda_array_B is {lambda_array_B}',flush=True)
    print(f'i.e. pop size_A is {A_popsize*N}',flush=True)
    print(f'i.e. pop size_B is {B_popsize*N}',flush=True)
    gamma=intensity
    print(f'gamma is {gamma}',flush=True)
    print(f'D is {D}',flush=True)
    print(f'ts is {ts}',flush=True)
    print(f'te is {te}',flush=True)
    print(f'N is {N}',flush=True)
    print(f'mu is {mu}',flush=True)
    if args.recomb_map is False:
        print(f'Error. Recomb map needed for this model={model}',flush=True)
        sys.exit()
    print(f'using chromosome {chrom}',flush=True)
    print(f'using recombination map: {recomb_map}',flush=True)
    rate_map = msprime.RateMap.read_hapmap(recomb_map)
    command_string = f'split_psc_220502(N=N,mu=mu,lambda_array_A=lambda_array_A,gamma=gamma,lambda_array_B=lambda_array_B,T=T*2*N,recombination_map=rate_map,simulation_model="hudson", \
                        verbosity=True,recordfullarg=recordfullarg)'
elif model=='split_psc_Bchange_220502':
    tm_true = Transition_Matrix_211220(D,spread_1,spread_2) # initialise transition matrix object
    T = tm_true.T # time array
    ts = 10
    te = 24
    A_popsize = np.ones(D)
    B_popsize = np.ones(D)

    A_popsize[0:4] = 3
    A_popsize[4:10] = 0.2
    A_popsize[10:13] = 1
    B_popsize[13:20] = 2
    A_popsize[te:te+3] = 1.5
    lambda_array_A = 1/A_popsize
    lambda_array_B = 1/B_popsize

    print(f'lambda_array_A is {lambda_array_A}',flush=True)
    print(f'lambda_array_B is {lambda_array_B}',flush=True)
    print(f'i.e. pop size_A is {A_popsize*N}',flush=True)
    print(f'i.e. pop size_B is {B_popsize*N}',flush=True)
    gamma=intensity
    print(f'gamma is {gamma}',flush=True)
    print(f'D is {D}',flush=True)
    print(f'ts is {ts}',flush=True)
    print(f'te is {te}',flush=True)
    print(f'N is {N}',flush=True)
    print(f'mu is {mu}',flush=True)
    if args.recomb_map is False:
        print(f'Error. Recomb map needed for this model={model}',flush=True)
        sys.exit()
    print(f'using chromosome {chrom}',flush=True)
    print(f'using recombination map: {recomb_map}',flush=True)
    rate_map = msprime.RateMap.read_hapmap(recomb_map)
    command_string = f'split_psc_220502(N=N,mu=mu,lambda_array_A=lambda_array_A,gamma=gamma,lambda_array_B=lambda_array_B,T=T*2*N,recombination_map=rate_map,simulation_model="hudson", \
                        verbosity=True,recordfullarg=recordfullarg)'
elif model=='split_psc_220504':
    tm_true = Transition_Matrix_211220(D,spread_1,spread_2) # initialise transition matrix object
    T = tm_true.T # time array
    ts = 10
    te = 24
    A_popsize = np.ones(D)
    B_popsize = np.ones(D)
    lambda_array_A = 1/A_popsize
    lambda_array_B = 1/B_popsize

    print(f'lambda_array_A is {lambda_array_A}',flush=True)
    print(f'lambda_array_B is {lambda_array_B}',flush=True)
    print(f'i.e. pop size_A is {A_popsize*N}',flush=True)
    print(f'i.e. pop size_B is {B_popsize*N}',flush=True)
    gamma=intensity
    print(f'gamma is {gamma}',flush=True)
    print(f'D is {D}',flush=True)
    print(f'ts is {ts}',flush=True)
    print(f'te is {te}',flush=True)
    print(f'N is {N}',flush=True)
    print(f'mu is {mu}',flush=True)
    if args.recomb_map is False:
        print(f'Error. Recomb map needed for this model={model}',flush=True)
        sys.exit()
    print(f'using chromosome {chrom}',flush=True)
    print(f'using recombination map: {recomb_map}',flush=True)
    rate_map = msprime.RateMap.read_hapmap(recomb_map)
    command_string = f'split_psc_220504(N=N,mu=mu,gamma=gamma,T=T*2*N,recombination_map=rate_map,simulation_model="hudson", \
                        verbosity=True,recordfullarg=recordfullarg)'
elif model=='halvingpop_220519': # (N,mu,p,L,T,simulation_model="hudson",verbosity=True)
    tm_true = Transition_Matrix_211220(D,spread_1,spread_2) # initialise transition matrix object
    T = tm_true.T # time array
    print(f'N is {N}')
    print(f'mu is {mu}')
    print(f'p is {p}')
    print(f'L is {L}')
    command_string = f'halvingpop_220519(N=N,mu=mu,p=p,L=L,T=T*2*N,simulation_model="hudson",verbosity=True)'
elif model=='triplingpop_220519': # (N,mu,p,L,T,simulation_model="hudson",verbosity=True)
    tm_true = Transition_Matrix_211220(D,spread_1,spread_2) # initialise transition matrix object
    T = tm_true.T # time array
    print(f'N is {N}')
    print(f'mu is {mu}')
    print(f'p is {p}')
    print(f'L is {L}')
    command_string = f'triplingpop_220519(N=N,mu=mu,p=p,L=L,T=T*2*N,simulation_model="hudson",verbosity=True)'
elif model=='triplingpop_221121': # (N,mu,p,L,T,simulation_model="hudson",verbosity=True)
    tm_true = Transition_Matrix_211220(D,spread_1,spread_2) # initialise transition matrix object
    T = tm_true.T # time array
    print(f'N is {N}')
    print(f'mu is {mu}')
    print(f'p is {p}')
    print(f'L is {L}')
    command_string = f'triplingpop_221121(N=N,mu=mu,p=p,L=L,T=T*2*N,simulation_model="hudson",verbosity=True)'
elif model=='popchanges_221113': # (N,mu,p,L,T,simulation_model="hudson",verbosity=True)
    tm_true = Transition_Matrix_211220(D,spread_1,spread_2) # initialise transition matrix object
    T = tm_true.T # time array
    print(f'N is {N}')
    print(f'mu is {mu}')
    print(f'p is {p}')
    print(f'L is {L}')
    command_string = f'popchanges_221113(N=N,mu=mu,p=p,L=L,T=T*2*N,simulation_model="hudson",verbosity=True)'
elif model=='quarteringpop_220519': # (N,mu,p,L,T,simulation_model="hudson",verbosity=True)
    tm_true = Transition_Matrix_211220(D,spread_1,spread_2) # initialise transition matrix object
    T = tm_true.T # time array
    print(f'N is {N}')
    print(f'mu is {mu}')
    print(f'p is {p}')
    print(f'L is {L}')
    command_string = f'quarteringpop_220519(N=N,mu=mu,p=p,L=L,T=T*2*N,simulation_model="hudson",verbosity=True)'
elif model=='constpop_220519': # (N,mu,p,L,T,simulation_model="hudson",verbosity=True)'
    tm_true = Transition_Matrix_211220(D,spread_1,spread_2) # initialise transition matrix object
    T = tm_true.T # time array
    print(f'N is {N}')
    print(f'mu is {mu}')
    print(f'p is {p}')
    print(f'L is {L}')
    command_string = f'constpop_220519(N=N,mu=mu,p=p,L=L,T=T*2*N,simulation_model="hudson",verbosity=True)'
elif model=='psc_europeanlike_220607':

    tm_true = Transition_Matrix_211220(D,spread_1,spread_2) # initialise transition matrix object
    T = tm_true.T # time array

    lambda_array = np.ones(D)
    lambda_array[0:6] = 0.3
    lambda_array[6:12] = 5
    lambda_array[12:20] = 0.6
    lambda_array[20:25] = 1
    lambda_array[25:30] = 0.7
    
    print(f'lambda_array is {lambda_array}',flush=True)
    print(f'i.e. pop sizes are {(1/lambda_array)*N}',flush=True)
    print(f'N is {N}',flush=True)
    print(f'L is {L}',flush=True)

    print(f'mu is {mu}',flush=True)
    print(f'p is {p}',flush=True)


    if args.mutation_map is False:
        print(f'Error. mutation map needed for this model={model}',flush=True)
        sys.exit()
    print(f'using chromosome {chrom}',flush=True)
    mu_mean = mu
    chrom_length = L
    alpha=mutation_map_alpha
    beta=mutation_map_beta
    change_rate=mutation_map_change_rate
    num_mu_domains = mutation_map_num_mu_domains 
    print(f'alpha={alpha};beta={beta};change_rate={change_rate}')
    num_mu_domains

    position, mus = generate_random_mutation_map(mu_mean=mu_mean,chrom_length=chrom_length,change_rate=change_rate,alpha=alpha,beta=beta,num_mu_domains=num_mu_domains)
    position[-1] = int(position[-1])
    mutation_rate_map = msprime.RateMap(
        position=position,
        rate=mus
    )
    # sliced_rate_map = rate_map.slice(0,1e+05)
    # sim,debug = psc_europeanlike_220427(N=N,mu=mu,T=T,recombination_map=sliced_rate_map,simulation_model="hudson", \
                        # verbosity=False,recordfullarg=recordfullarg)
    # RAA, _ = debug.coalescence_rate_trajectory(T*2*N, {"A": 2})
    # print_iCR=True
    # if print_iCR:
        # print(f'1/(2*RAA) =\n{1/(2*RAA)}',flush=True)
    command_string = f'psc_europeanlike_220607(N=N,lambda_array=lambda_array,mu=mu,mutation_map=mutation_rate_map,p=p,T=T*2*N,simulation_model="hudson", \
                        verbosity=True,recordfullarg=recordfullarg)'
elif model=='pop_simple_split_220712':
    # I use N as haploid here
    print(f'D is {D}')
    print(f'spread_1 is {spread_1}')
    print(f'spread_2 is {spread_2}')
    print(f'N is {N}')
    print(f'mu is {mu}')
    print(f'p is {p}')
    print(f'L is {L}')
    ts = start_index
    te = end_index
    print(f'ts is {ts}')
    print(f'te is {te}')
    print(f'gamma is {intensity}')
    
    lambda_array = np.ones(D)
    pop_sizes = (1/(lambda_array))*(N)
    print(f'pop sizes are {pop_sizes}')
    T_sim_coalescent = time_intervals(D,spread_1,spread_2)
    T_sim_gens = T_sim_coalescent*2*N
    print(f'T_gens is {T_sim_gens}')
    command_string = f'pop_simple_split_220712(N,pop_sizes,mu,p,L,T_sim_gens,start_index,end_index,intensity,simulation_model=sim_model, \
                        return_debug=True,recordfullarg=recordfullarg,verbosity=verbosity_present)'
    sim, debug_cr = pop_simple_split_220712(N,pop_sizes,mu,p,10,T_sim_gens,start_index,end_index,intensity,simulation_model=sim_model, \
                        return_debug=True,recordfullarg=recordfullarg,verbosity=verbosity_present)


    RAA, _ = debug_cr.coalescence_rate_trajectory(T_sim_gens, {"A": 2})
    print(f'coalescent rate is \n {1/RAA[0:-1]}')


elif model=='pop_simpleconst_220712':
    # I use N as haploid here
    print(f'D is {D}')
    print(f'spread_1 is {spread_1}')
    print(f'spread_2 is {spread_2}')
    print(f'N is {N}')
    print(f'mu is {mu}')
    print(f'p is {p}')
    print(f'L is {L}')
    
    lambda_array = np.ones(D)
    pop_sizes = (1/(lambda_array))*(N)
    print(f'pop sizes are {pop_sizes}')
    T_sim_coalescent = time_intervals(D,spread_1,spread_2)
    T_sim_gens = T_sim_coalescent*2*N
    print(f'T_gens is {T_sim_gens}')
    

    command_string = f'pop_simpleconst_220712(N,pop_sizes,mu,p,L,T_sim_gens,simulation_model=sim_model, \
                        return_debug=True,recordfullarg=recordfullarg,verbosity=verbosity_present)'
    sim, debug_cr = pop_simpleconst_220712(N,pop_sizes,mu,p,10,T_sim_gens,simulation_model=sim_model, \
                        return_debug=True,recordfullarg=recordfullarg,verbosity=verbosity_present)
    RAA, _ = debug_cr.coalescence_rate_trajectory(T_sim_gens, {"A": 2})
    print(f'coalescent rate is \n {1/RAA[0:-1]}')


elif model=='pop_split_nonafrican_icr_220712':
    
    print(f'D is {D}')
    print(f'spread_1 is {spread_1}')
    print(f'spread_2 is {spread_2}')
    print(f'N is {N}')
    print(f'mu is {mu}')
    print(f'p is {p}')
    print(f'L is {L}')
    print(f'ts is {start_index}')
    print(f'te is {end_index}')
    ts = start_index
    te = end_index
    
    lambda_array = np.ones(D)
    lambda_array[0:3] = 0.6
    lambda_array[3:11] = 7
    lambda_array[11:20] = 1
    lambda_array[20:28] = 1
    lambda_array[28:33] = 0.57
    psc_nonafrican_icr_split_like_220713 = (1/lambda_array)*(N)
    pop_sizes = psc_nonafrican_icr_split_like_220713
    T_sim_coalescent = time_intervals(D,spread_1,spread_2)
    T_sim_gens = T_sim_coalescent*2*N
    print(f'T_gens is {T_sim_gens}')
    command_string = f'pop_split_nonafrican_icr_220712(N,pop_sizes,mu,p,L,T_sim_gens,ts,te,simulation_model=sim_model, \
                    return_debug=True,recordfullarg=recordfullarg,verbosity=verbosity_present)'
    sim, debug_cr = pop_split_nonafrican_icr_220712(N,pop_sizes,mu,p,10,T_sim_gens,ts,te,simulation_model=sim_model, \
                    return_debug=True,recordfullarg=recordfullarg,verbosity=verbosity_present)
    RAA, _ = debug_cr.coalescence_rate_trajectory(T_sim_gens, {"A": 2})
    print(f'coalescent rate is \n {1/RAA[0:-1]}')


elif model=='pop_nonafrican_icr_220712_matching_psc':
    
    print(f'D is {D}')
    print(f'spread_1 is {spread_1}')
    print(f'spread_2 is {spread_2}')
    print(f'N is {N}')
    print(f'mu is {mu}')
    print(f'p is {p}')
    print(f'L is {L}')
    print(f'ts is {start_index}')
    print(f'te is {end_index}')
    ts = start_index
    te = end_index

    lambda_array = np.ones(D)
    lambda_array[0:3] = 0.6
    lambda_array[3:11] = 7
    lambda_array[11:20] = 1
    lambda_array[20:28] = 1
    lambda_array[28:33] = 0.57
    T_sim_coalescent = time_intervals(D,spread_1,spread_2)
    T_sim_gens = T_sim_coalescent*2*N
    print(f'T_gens is {T_sim_gens}')

    psc_nonafrican_icr_split_like_220713 = (1/lambda_array)*(N)
    L_new = 10
    sim_split_nonafrican_icr,debug_split_nonafrican_icr = pop_split_nonafrican_icr_220712(N,psc_nonafrican_icr_split_like_220713,mu,p,L_new,T_sim_gens,ts,te,simulation_model=sim_model,return_debug=True,recordfullarg=False,verbosity=False)
    RAA_split_nonafrican_icr, _ = debug_split_nonafrican_icr.coalescence_rate_trajectory(T_sim_gens, {"A": 2})
    psc_nonafrican_icr_220713_matching_psc=(1/RAA_split_nonafrican_icr[0:-1])/2

    command_string = f'pop_nonafrican_icr_220712_matching_psc(N,psc_nonafrican_icr_220713_matching_psc,mu,p,L,T_sim_gens,simulation_model=sim_model, \
                return_debug=True,recordfullarg=recordfullarg,verbosity=verbosity_present)'
    sim, debug_cr = pop_nonafrican_icr_220712_matching_psc(N,psc_nonafrican_icr_220713_matching_psc,mu,p,L_new,T_sim_gens,simulation_model=sim_model, \
                return_debug=True,recordfullarg=recordfullarg,verbosity=verbosity_present)
    RAA, _ = debug_cr.coalescence_rate_trajectory(T_sim_gens, {"A": 2})
    print(f'coalescent rate is \n {1/RAA[0:-1]}')
elif model=='pop_split_custom_lambda_220721':
    
    print(f'D is {D}')
    print(f'spread_1 is {spread_1}')
    print(f'spread_2 is {spread_2}')
    print(f'N is {N}')
    print(f'mu is {mu}')
    print(f'p is {p}')
    print(f'L is {L}')
    print(f'ts is {start_index}')
    print(f'te is {end_index}')
    print(f'lambdas_A is {lambdas_A}')
    print(f'lambdas_B is {lambdas_B}')
    print(f'gamma is {intensity}')
    gamma_sim = intensity
    T_sim_coalescent = time_intervals(D,spread_1,spread_2)
    T = T_sim_coalescent
    T_sim_gens = T_sim_coalescent*2*N
    print(f'T_gens is {T_sim_gens}')

    command_string = f'pop_split_custom_lambda_220721(N,lambdas_A,lambdas_B,mu,p,L,T_sim_gens,start_index,end_index,gamma_sim,simulation_model="hudson",verbosity=False,recordfullarg=recordfullarg)'
    
    sim, debug_cr = pop_split_custom_lambda_220721(N,lambdas_A,lambdas_B,mu,p,10,T_sim_gens,start_index,end_index,gamma_sim,simulation_model="hudson",verbosity=True,recordfullarg=False)
    
    RAA, _ = debug_cr.coalescence_rate_trajectory(T_sim_gens, {"A": 2})
    print(f'coalescent rate is \n {1/RAA[0:-1]}')
elif model=='pop_custom_lambda_220721':
    
    print(f'D is {D}')
    print(f'spread_1 is {spread_1}')
    print(f'spread_2 is {spread_2}')
    print(f'N is {N}')
    print(f'mu is {mu}')
    print(f'p is {p}')
    print(f'L is {L}')
    print(f'(not used) ts is {start_index}')
    print(f'(not used) te is {end_index}')
    print(f'lambdas_A is {lambdas_A}')
    print(f'(not used) lambdas_B is {lambdas_B}')
    print(f'(not used) gamma is {intensity}')
    gamma_sim = intensity
    T_sim_coalescent = time_intervals(D,spread_1,spread_2)
    T = T_sim_coalescent
    T_sim_gens = T_sim_coalescent*2*N
    print(f'T_gens is {T_sim_gens}')

    command_string = f'pop_custom_lambda_220721(N,lambdas_A,lambdas_B,mu,p,L,T_sim_gens,start_index,end_index,gamma_sim,simulation_model="hudson",verbosity=False,recordfullarg=recordfullarg)'
    
    sim, debug_cr = pop_custom_lambda_220721(N,lambdas_A,lambdas_B,mu,p,10,T_sim_gens,start_index,end_index,gamma_sim,simulation_model="hudson",verbosity=True,recordfullarg=False)
    
    RAA, _ = debug_cr.coalescence_rate_trajectory(T_sim_gens, {"A": 2})
    print(f'coalescent rate is \n {1/RAA[0:-1]}')
elif model=='pop_cont_model_220913':
    print(f'D is {D}')
    print(f'spread_1 is {spread_1}')
    print(f'spread_2 is {spread_2}')
    print(f'N is {N}')
    print(f'mu is {mu}')
    print(f'p is {p}')
    print(f'L is {L}')
    print(f'ts is {start_index}') 
    print(f'te is {end_index}')
    print(f'lambdas_A is {lambdas_A}') 
    print(f'lambdas_B is {lambdas_B}')
    print(f'recordfullarg is {recordfullarg}')
    print(f'm is {intensity}')
    m_sim = intensity
    T_sim_coalescent = time_intervals(D,spread_1,spread_2)
    T = T_sim_coalescent
    T_sim_gens = T_sim_coalescent*2*N
    print(f'T_gens is {T_sim_gens}')
    command_string = f'pop_cont_model_220913(N,lambdas_A,mu,m_sim,p,L,T_sim_gens,start_index,end_index,simulation_model=sim_model,return_debug=True,recordfullarg=recordfullarg,verbosity=True)'
    
    sim, debug_cr = pop_cont_model_220913(N,lambdas_A,mu,m_sim,p,10,T_sim_gens,start_index,end_index,simulation_model=sim_model,return_debug=True,recordfullarg=False,verbosity=True)
    RAA, _ = debug_cr.coalescence_rate_trajectory(T_sim_gens, {"A": 2})

    print(f'coalescent rate is \n {1/RAA[0:-1]}')
elif model=='pop_split_model_220913':
    print(f'D is {D}')
    print(f'spread_1 is {spread_1}')
    print(f'spread_2 is {spread_2}')
    print(f'N is {N}')
    print(f'mu is {mu}')
    print(f'p is {p}')
    print(f'L is {L}')
    print(f'ts is {start_index}') 
    print(f'te is {end_index}')
    print(f'lambdas_A is {lambdas_A}') 
    print(f'lambdas_B is {lambdas_B}')
    print(f'recordfullarg is {recordfullarg}')
    print(f'gamma is {intensity}')
    gamma_sim = intensity
    T_sim_coalescent = time_intervals(D,spread_1,spread_2)
    T = T_sim_coalescent
    T_sim_gens = T_sim_coalescent*2*N
    print(f'T_gens is {T_sim_gens}')
    command_string = f'pop_split_model_220913(N,lambdas_A,mu,gamma_sim,p,L,T_sim_gens,start_index,end_index,simulation_model=sim_model,return_debug=True,recordfullarg=recordfullarg,verbosity=True)'
    sim, debug_cr = pop_split_model_220913(N,lambdas_A,mu,gamma_sim,p,10,T_sim_gens,start_index,end_index,simulation_model=sim_model,return_debug=True,recordfullarg=False,verbosity=True)
    # flag220913

    RAA, _ = debug_cr.coalescence_rate_trajectory(T_sim_gens, {"A": 2})

    print(f'coalescent rate is \n {1/RAA[0:-1]}')
# elif model=='psc_europeanlike_220607_nomutationmap':
    
#     tm_true = Transition_Matrix_211220(D,spread_1,spread_2) # initialise transition matrix object
#     T = tm_true.T # time array

#     lambda_array = np.ones(D)
#     lambda_array[0:6] = 0.3
#     lambda_array[6:12] = 5
#     lambda_array[12:20] = 0.6
#     lambda_array[20:25] = 1
#     lambda_array[25:30] = 0.7
    
#     print(f'lambda_array is {lambda_array}',flush=True)
#     print(f'i.e. pop sizes are {(1/lambda_array)*N}',flush=True)
#     print(f'N is {N}',flush=True)
#     print(f'L is {L}',flush=True)

#     print(f'mu is {mu}',flush=True)
#     print(f'p is {p}',flush=True)


#     print(f'using chromosome {chrom}',flush=True)
#     # sliced_rate_map = rate_map.slice(0,1e+05)
#     # sim,debug = psc_europeanlike_220427(N=N,mu=mu,T=T,recombination_map=sliced_rate_map,simulation_model="hudson", \
#                         # verbosity=False,recordfullarg=recordfullarg)
#     # RAA, _ = debug.coalescence_rate_trajectory(T*2*N, {"A": 2})
#     # print_iCR=True
#     # if print_iCR:
#         # print(f'1/(2*RAA) =\n{1/(2*RAA)}',flush=True)
#     command_string = f'psc_europeanlike_220607_nomutationmap(N=N,L=L,lambda_array=lambda_array,mu=mu,p=p,T=T*2*N,simulation_model="hudson", \
#                         verbosity=True,recordfullarg=recordfullarg)'
else:
    print(f'model={model} not recognised. Aborting',flush=True)
    sys.exit()
hets_all = np.array([],dtype=int)

print(f'command string is {command_string}',flush=True)
if args.write_mhs is True: # write mhs files
    if mutation_map is not True:
        if command_string!='':
            exec(f'sim,debug = {command_string}')
        hets = np.array([int(np.rint(var.site.position)) for var in sim.variants() if var.genotypes[0]!=var.genotypes[1]])
        
    elif mutation_map is True:
        exec(f'msim_mutmap, msim_nomutmap, debug = {command_string}')
        hets_mutmap = np.array([int(np.rint(var.site.position)) for var in msim_mutmap.variants() if var.genotypes[0]!=var.genotypes[1]])
        hets_nomutmap = np.array([int(np.rint(var.site.position)) for var in msim_nomutmap.variants() if var.genotypes[0]!=var.genotypes[1]])
    if output_file_mhs is not None:
        filename = output_file_mhs
    else:
        if output_path_mhs[-1] != '/':
            output_path_mhs = output_path_mhs + '/' 
        filename = output_path_mhs + f'sample{sample}_chrom{chrom}.mhs'
    print(f'saving mhs file to {filename}',flush=True)
    if write_output is True and recomb_map is not False:
        write_mhs(hets,filename,chrom,rate_map)
    elif write_output is True and recomb_map is False and mutation_map is False:
        write_mhs(hets,filename,chrom)
    elif write_output is True and mutation_map is True: # save mutation map used
        write_mhs(hets_mutmap,filename,chrom)
        filename_nomutmap = output_path_mhs_nomutmap + f'sample{sample}_chrom{chrom}.mhs'
        write_mhs(hets_nomutmap,filename_nomutmap,chrom)
        print(f'saving mhs_nomutmap file to {filename_nomutmap}',flush=True)

        np_rate_map = np.array([position[1:],mus]).T # left column is the right boundary of segment (starting at 0); right column is the rate for that segment
        mutmap_filename = output_path_mhs + f'sample{sample}_chrom{chrom}_ratemap.txt'
        np.savetxt(mutmap_filename,np_rate_map)
        tmrca_data = get_coal_data(msim_nomutmap)
        np.savetxt(output_tmrca,tmrca_data)
        print(f'saved mutation rate map to {mutmap_filename}')
        print(f'saved tmrca_data to {output_tmrca}')

    else:
        tmrca_data = get_coal_data(sim)
        np.savetxt(output_tmrca,tmrca_data)
        print(f'saved tmrca_data to {output_tmrca}')

else:
    # for j in range(0,iterations):
    j=0

    while j<iterations:
        print(f'On iteration {j} out of {iterations}',flush=True)
        exec(f'sim,debug = {command_string}')
        hets = np.array([int(np.rint(var.site.position)) for var in sim.variants() if var.genotypes[0]!=var.genotypes[1]])
        hets = hets + int(j)*L
        hets_all = np.concatenate((hets_all,hets))
        tmrca_data = get_coal_data(sim)
        if lineage_path is False:
            if j==0:
                tmrca_data_all = tmrca_data
            else:
                tmrca_data[:,0] = tmrca_data[:,0] + 1 + tmrca_data_all[:,0][-1]
                tmrca_data[:,1] = tmrca_data[:,1] + tmrca_data_all[:,1][-1]
                tmrca_data_all = np.concatenate((tmrca_data_all,tmrca_data),axis=0)
        elif lineage_path is True:
            path_data = get_lineage_data(sim,T*2*N,start_index,end_index)
            if np.min(path_data)==-1:
                print(f'skipping iteration {j}',flush=True)
                continue
            tmrca_data = combine_coal_data_lineage(tmrca_data,path_data)
            if j==0:
                tmrca_data_all = tmrca_data
            elif 'tmrca_data_all' not in locals():
                tmrca_data_all = tmrca_data
            else:
                tmrca_data[:,0] = tmrca_data[:,0] + 1 + tmrca_data_all[:,0][-1]
                tmrca_data[:,1] = tmrca_data[:,1] + tmrca_data_all[:,1][-1]
                tmrca_data_all = np.concatenate((tmrca_data_all,tmrca_data),axis=0)
            np.savetxt(output_tmrca,tmrca_data_all)
                # print(f'shape of tmrca_data is {tmrca_data.shape}')
            
        j += 1
        del sim, debug

    for k in range(1,4):
        exec(f'hets{k} = hets_all[np.where(hets_all<seq_lengths_int[{k-1}])]')

    if write_output is True:

        
        if dont_save_het_data is False:
            np.savetxt(output_hets,hets_all)
            print(f'Saving all het data to {output_hets}')
            for k in range(1,4):
                exec(f'dest = output_hets{k}') 
                exec(f'np.savetxt(dest,hets{k})')
                print(f'Saving het{k} data to {dest}',flush=True)

        np.savetxt(output_tmrca,tmrca_data_all)
        # print(f'Saving tmrca_data to {tmrca_path}')
        print(f'Saving tmrca_data to {output_tmrca}',flush=True)
    else:
        print(f'\tNot writing output files as write_output is {write_output}',flush=True)

    print(f'num_hets is {len(hets_all)}',flush=True)
    print(f'num_recombs is {tmrca_data_all.shape[0]}',flush=True)

sys.exit()
