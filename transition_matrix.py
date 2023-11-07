import numpy as np
import pdb
import sys
import time
from numba import njit, jit

@njit
def time_intervals(D,spread_1,spread_2,final_T_factor=None):
    # D is int for number of time intervals
    # spread_1 and spread_2 are floats which control spread of time intevals
    # final_T_factor is boolean, which if given decides how much bigger T[-1] is than T[-2]. If None (default) then T[-1] is written according to the equation 
    T = np.zeros(D+1)
    for i in range(0,D): 
        T[i+1] = spread_1*np.exp( (i/D)*np.log(1 + spread_2/spread_1) - 1)
    if final_T_factor is not None:
        T[-1] = T[-2]*final_T_factor    
    return T

# difference between two intervals
@njit
def delta(T,index):
    # T is array of D+1 time intervals in coalescent time
    # index is int of lower index
    delt = T[index+1] - T[index]
    return delt

# find index in T of given times
@njit
def index_finder(T,t1,t2):
    if t1 > t2:
        # print('problem in index_finder. t1 = {} > t2 = {}\nAborting.'.format(t1,t2))
        print('problem in index_finder. t1 > t2\nAborting')
    diff_t1 = T-t1 # difference in T and t1
    t1_upper = np.min(diff_t1[np.where(diff_t1>0)[0]]) # what is smallest positive number
    t1_upper_index = np.where(diff_t1==t1_upper)[0][0] #  what index does this have in T
    diff_t2 = T-t2 # difference in T and t2
    t2_lower = np.max(diff_t2[np.where(diff_t2<=0)[0]]) # what is biggest nonpositive number
    t2_lower_index = np.where(diff_t2==t2_lower)[0][0] # what index does this have in T
    return t1_upper_index,t2_lower_index

# structured G
# def G_0(T_S,T_E,gamma,lambda_A,lambda_B,t):
#     # probability of not coalescing from 0 to t with strutured parameters
#     # T_S and T_E are indices (ints)
#     # gamma is float 
#     # lambda_A is array of size D
#     # lambda_B is array of size D
#     if T_S==None or t<T_E:
#         G = L(0,t,lambda_A)
#     elif (t>=T_S and t<T_E):
#         G = L(0,T_s,lambda_A)*( ((1-gamma)**2)*L(T_S,t,lambda_A) + ((gamma)**2)*self.L(T_S,t,lambda_B))
#     elif (t>=T_e):
#         G = L(0,T_s,lambda_A)*( ((1-gamma)**2)*L(T_s,T_e,lambda_A) + ((gamma)**2)*L(T_s,T_e,lambda_B)+2*gamma*(1-gamma) )*L(T_e,t,lambda_A)
#     else:
#         print('misunderstanding in G_0()!')
#     return G
    

# probability of not coalescing between t1 and t2 (for changing population size; no structure here)
@njit
def L(T,t1,t2,lambd):
    # T is array of D+1 time intervals in coalescent time
    # t1 and t2 are floats for time in coalescent time
    # find index for t1,t2
    # all time should be in coalescent time; T should be np
    t1_upper_index, t2_lower_index = index_finder(T,t1,t2) # get index of t1 and t2 wrt T
    if t1_upper_index==(t2_lower_index+1): # if in same interval
        # print('t1_upper_index is {}, t2_lower_index is {}, lambda[t1_lower_index] is {}, lambda[t2_upper_index] is {}'.format(t1_upper_index,t2_lower_index,lambd[t1_upper_index-1],lambd[t2_lower_index]))
        L = np.exp(-(t2-t1)*lambd[t2_lower_index])
    else:
        summation = 0 # if in adjacent intervals, summation should be 0
        for i in range(t1_upper_index,t2_lower_index): # sum from t1_upper_index to (t2_lower_index -1)
            summation += lambd[i]*delta(T,i) 
        L = np.exp( - (T[t1_upper_index]-t1)*lambd[t1_upper_index-1] - summation - (t2-T[t2_lower_index])*lambd[t2_lower_index])
    return L

def summation(D,T,lambd,factor):
    # D is int for number of time intervals
    # T is array for time interval boundaries in coalescent time
    # lambd is array of lambdas 
    # factor is exponent
    summations = np.zeros(D)
    for k in range(1,D):
        for j in range(0,k):
            sum_ += (1/(factor*lambd[j]))*(1-np.exp(-factor*lambd[j]*delta(T,j)))
        summations[k] = sum_ 
    return summations

# structured e_beta
@njit
def e_beta(D,T,lambda_A,lambda_B,T_S,T_E,gamma,int,midpoint_transitions=False):
    # D is int for number of time states
    # T is array of time interval boundaries in coalescent units
    # lambda_A is array of inverse scaled population size parameters for population A
    # lambda_B is array of inverse scaled population size parameters for population B
    # T_S is float for rejoin time in coalescence units
    # T_E is float for split time in coalescence units
    # gamma is float for split fraction
    # midpoint_transitions is boolean, if True then set the expected coalescence time as midpoint (not recommended)

    # function for expected time in beta, written <t_beta> in equations
    if T_S==None or T[int]<T_S:
        numerator = lambda_A[int]*T[int]+1 - (lambda_A[int]*T[int+1]+1)*np.exp(-delta(T,int)*lambda_A[int])
        denominator = lambda_A[int]*(1 -  np.exp(-delta(T,int)*lambda_A[int]) )
        new_e = numerator/denominator
        # print(f'new_e is \t{new_e}\n')

    elif (T[int]>=T_S) and (T[int]< T_E):
        denominator = L(T,0,T_S,lambda_A)*(
            ((1-gamma)**2)*L(T,T_S,T[int],lambda_A)*(1-np.exp(-delta(T,int)*lambda_A[int])) + 
            ((gamma)**2)*L(T,T_S,T[int],lambda_B)*(1-np.exp(-delta(T,int)*lambda_B[int]))
        )                  
        numerator = L(T,0,T_S,lambda_A)*(
            ((1-gamma)**2)*L(T,T_S,T[int],lambda_A)*(1/lambda_A[int])*(
            lambda_A[int]*T[int]+1 - (lambda_A[int]*T[int+1]+1)*np.exp(-delta(T,int)*lambda_A[int])
                                                        )
            + (gamma**2)*L(T,T_S,T[int],lambda_B)*(1/lambda_B[int])*(
            lambda_B[int]*T[int]+1 - (lambda_B[int]*T[int+1]+1)*np.exp(-delta(T,int)*lambda_B[int])
                                                        )
        )
    elif T[int]>=T_E:
        denominator = L(T,0,T_S,lambda_A)*(
            (((1-gamma)**2)*L(T,T_S,T_E,lambda_A) + (gamma**2)*L(T,T_S,T_E,lambda_B) + 2*gamma*(1-gamma))*(
                L(T,T_E,T[int],lambda_A)*(1-np.exp(-delta(T,int)*lambda_A[int])) )
        )
        numerator = L(T,0,T_S,lambda_A)*(
            (((1-gamma)**2)*L(T,T_S,T_E,lambda_A) + (gamma**2)*L(T,T_S,T_E,lambda_B) + 2*gamma*(1-gamma))*(
                L(T,T_E,T[int],lambda_A)*(1/lambda_A[int])*(
                    lambda_A[int]*T[int]+1 - (lambda_A[int]*T[int+1]+1)*np.exp(-delta(T,int)*lambda_A[int])
                )
            )
        )            
    else: 
        print('misunderstanding in e_beta()!')
    e = numerator/denominator
    if midpoint_transitions==True:
        e = (T[int] + T[int+1])/2 # take midpoint
    return e

# check expected times in inteval beta                         
def e_beta_check(D,T,lambda_A,lambda_B,T_S,T_E,gamma):
    # D is int for number of time states
    # T is array of time interval boundaries in coalescent units
    # lambda_A is array of inverse scaled population size parameters for population A
    # lambda_B is array of inverse scaled population size parameters for population B
    # T_S is float for rejoin time in coalescence units
    # T_E is float for split time in coalescence units
    # gamma is float for split fraction

    # check that e_beta() is behaving as expected
    # We should have T[int] <= e_beta(int) < T[int+1]
    check = 0
    print_ = False
    for int in range(0,len(T)-1):
        if print_:
            print('\nint is {};len(T) is {}'.format(int,len(T)))
        e_bet = e_beta(D,T,lambda_A,lambda_B,T_S,T_E,gamma,int)
        # e_bet = self.e_betas[int]
        if e_bet<T[int] or e_bet>T[int+1]:
            print('\nError in e_beta()')
            print('T_s is {} and T_e is {}'.format(T_S,T_E))
            print('T[int] is {}; e_beta(int) is {}; T[int+1] is {}'.format(T[int],e_bet,T[int+1]))
            check = 1
        else:
            print_ = False
            if print_:
                print('performing well. T[int] is {}; e_beta(int) is {}; T[int+1] is {}'.format(T[int],e_bet,T[int+1]))
    if check != 0:
        print('e_beta() is NOT performing satisfactorily')
    # else:
    #     print('e_beta() is performing satisfactorily')
    return None


def check_all(D,T,lambda_A,lambda_B,T_S,T_E,gamma): # check parameters are all acceptable
    # D is int for number of time states
    # T is array of time interval boundaries in coalescent units
    # lambda_A is array of inverse scaled population size parameters for population A
    # lambda_B is array of inverse scaled population size parameters for population B
    # T_S is float for rejoin time in coalescence units
    # T_E is float for split time in coalescence units
    # gamma is float for split fraction
    

    if type(D)!=int:
        print(f'D={D} must be int')
    if type(T)!=np.ndarray:
        print(f'T={T} must be np.ndarray')
    if type(lambda_A)!=np.ndarray:
        print(f'lambda_A={lambda_A} must be np.ndarray')

    # TODO check parameters are in right numerical range, (eg every element in lambda_A is bigger than 0)

    check=0
    try:
        list_of_possible_parameters = [list(lambda_B),T_S,T_E,gamma]  # the set {lambda_B,T_s,T_e,gamma} should either all exist or not
    except:
        list_of_possible_parameters = [lambda_B,T_S,T_E,gamma]  # the set {lambda_B,T_s,T_e,gamma} should either all exist or not
    if None in list_of_possible_parameters: # if there is one None, they should all be none
        for item in list_of_possible_parameters:
            if item is not None:
                print('Problem! The following parameters {lambda_B,T_s,T_e,gamma} should all be None or all defined. \nAborting.')
                check=1
                # sys.exit()
    else: # {lambda_B,T_s,T_e,gamma} are all defined
        if len(lambda_A)!=len(lambda_B):
            print('Problem! lambda_A is of length {} and lambda_B is of length {}. They should be the same.\nAborting.'.format(len(lambda_A),len(lambda_B)))
            check=1
            # sys.exit()
        if D!=len(lambda_A) or D!=(len(T)-1):
            print('Problem! lambda_A is of length {} and D is {} and T is of length {}. \nlambda_A length should equal D. Length of T should be one more.\nAborting.'.format(len(lambda_A),len(D),len(self.T)))
            check=1
            # sys.exit()            
        if T_S >= T_E:
            print('Problem! T_s = {} >= T_e = {}. T_s should be smaller than T_e'.format(T_S,T_E))
            check=1
            # sys.exit()
        if T_S<0 or T_E<0:
            print('Problem! T_s = {}, T_e = {}. They should be bigger than 0'.format(T_S,T_E))
            check=1
            # sys.exit()

        if gamma > 1 or gamma < 0:
            check=1
            print('Problem! gamma = {}, it should be between 0 and 1 '.format(T_E))
            # sys.exit() # this should NOT be commented out, but for a peculiar reason in Powell it remains
    e_beta_check(D,T,lambda_A,lambda_B,T_S,T_E,gamma)
    # print('Everything is good.')
    return check

@njit
def L_precomputations(D,T,lambda_A,lambda_B,T_S_li,T_E_li,gamma,e_betas):
    # D is int for number of time states
    # T is array of time interval boundaries in coalescent units
    # lambda_A is array of inverse scaled population size parameters for population A
    # lambda_B is array of inverse scaled population size parameters for population B
    # T_S_li is int for index of rejoin time in array T
    # T_E_li is int for index of split time in array T
    # T_S is float for rejoin time in coalescence units
    # T_E is float for split time in coalescence units
    # gamma is float for split fraction
    # e_betas is array of expected coalescence times
    
    # pre computations for L function
    L_s_A = np.zeros((D,D)) # for L() where t1 is s (which does not align with interval boundary) in pop A
    L_s_B = np.zeros((D,D)) # for L() where t1 is s (which does not align with interval boundary) in pop B
    L_A = np.zeros((D,D)) # for L() where t1 and t2 align with boundaries in pop A
    L_B = np.zeros((D,D)) # for L() where t1 and t2 align with boundaries in pop B
    
    L_Tse_s_A = np.zeros((D,D)) # for L() where t2 is s (which does not align with interval boundary) in pop A
    L_Tse_s_B = np.zeros((D,D)) # for L() where t2 is s (which does not align with interval boundary) in pop B
    
    # L(T_S,s,lambda_A)**2

    for beta in range(0,D):
        s = e_betas[beta] # s is the "expected time" in interval beta
        if T_S_li!=None:          
    
            T_S = T[T_S_li]
            T_E = T[T_E_li]

            # if beta>=T_S_li and beta<T_E_li:
            if beta<T_S_li:
                L_Tse_s_A[beta,beta] = L(T,T[beta],s,lambda_A)
                L_Tse_s_B[beta,beta] = L(T,T[beta],s,lambda_B)
            if beta>=T_S_li:
                L_Tse_s_A[T_S_li,beta] = L(T,T_S,s,lambda_A)
                L_Tse_s_B[T_S_li,beta] = L(T,T_S,s,lambda_B)
                L_Tse_s_A[beta,beta] = L(T,T[beta],s,lambda_A)
                L_Tse_s_B[beta,beta] = L(T,T[beta],s,lambda_B)                
                L_A[T_S_li,beta] = L(T,T_S,T[beta],lambda_A)
                L_B[T_S_li,beta] = L(T,T_S,T[beta],lambda_B)
            # elif beta>=T_E_li:
            if beta>=T_E_li:
                L_Tse_s_A[T_E_li,beta] = L(T,T_E,s,lambda_A)
                L_Tse_s_B[T_E_li,beta] = L(T,T_E,s,lambda_B)
                L_Tse_s_A[beta,beta] = L(T,T[beta],s,lambda_A)
                L_Tse_s_B[beta,beta] = L(T,T[beta],s,lambda_B)
                L_A[T_E_li,beta] = L(T,T_E,T[beta],lambda_A)
                L_B[T_E_li,beta] = L(T,T_E,T[beta],lambda_B)

        for alpha in range(beta+1,D):
            L_s_A[beta,alpha] = L(T,s,T[alpha],lambda_A)
            L_Tse_s_A[beta,beta] = L(T,T[beta],s,lambda_A)
            if T_S_li!=None:
                L_s_B[beta,alpha] = L(T,s,T[alpha],lambda_B)
                L_Tse_s_B[beta,beta] = L(T,T[beta],s,lambda_B)
    return L_s_A,L_s_B,L_A,L_B,L_Tse_s_A,L_Tse_s_B

# (Pdb) K(D,T,lambda_A,0,beta,2)
# (Pdb) K(D,T,lambda_A,0,T_S_li,2)
# (Pdb) K(D,T,lambda_A,T_S_li,beta,2)
# (Pdb) K(D,T,lambda_A,T_E_li,beta,2)
# (Pdb) K(D,T,lambda_A,0,beta,2)

# upper diagonal
@njit
def upper_Q(D,T,lambda_A,lambda_B,T_S_li,T_E_li,gamma,e_betas,Q,L_s_A,L_s_B,L_A,L_B,L_Tse_s_A,L_Tse_s_B,K_array):
    # D is int for number of time states
    # T is array of time interval boundaries in coalescent units
    # lambda_A is array of inverse scaled population size parameters for population A
    # lambda_B is array of inverse scaled population size parameters for population B
    # T_S_li is int for index of rejoin time in array T
    # T_E_li is int for index of split time in array T
    # T_S is float for rejoin time in coalescence units
    # T_E is float for split time in coalescence units
    # gamma is float for split fraction
    # e_betas is array of expected coalescence times
    # Q is transitions matrix 
    # L_s_A is array for L() where t1 is s (which does not align with interval boundary) in pop A
    # L_s_B is array for L() where t1 is s (which does not align with interval boundary) in pop B
    # L_A is array for L() where t1 and t2 align with boundaries in pop A
    # L_B is array for L() where t1 and t2 align with boundaries in pop B
    # L_Tse_s_A is array for L() where t2 is s (which does not align with interval boundary) in pop A
    # L_Tse_s_B is array for L() where t2 is s (which does not align with interval boundary) in pop B
    # K_array is an array for the pre computed values of the K function. Iindices correspond to [row,column,pop,factor]

    if T_S_li==None:
        T_S = None
        T_E = None
    else:
        T_S = T[T_S_li]
        T_E = T[T_E_li]
    

    # structured pre computations
    if T_S_li!=None: 

        prob_aa_case10 = (((1-gamma)**2)*L(T,T_S,T_E,lambda_A))/(((1-gamma)**2)*L(T,T_S,T_E,lambda_A) + ((gamma)**2)*L(T,T_S,T_E,lambda_B) +2*gamma*(1-gamma))
        prob_bb_case10 = (((gamma)**2)*L(T,T_S,T_E,lambda_B))/(((1-gamma)**2)*L(T,T_S,T_E,lambda_A) + ((gamma)**2)*L(T,T_S,T_E,lambda_B) +2*gamma*(1-gamma))
        prob_ab_case10 = (2*gamma*(1-gamma))/(((1-gamma)**2)*L(T,T_S,T_E,lambda_A) + ((gamma)**2)*L(T,T_S,T_E,lambda_B) +2*gamma*(1-gamma))

        prob_aa_case7 = np.zeros(D)
        prob_bb_case7 = np.zeros(D)
        for k in range(0,D):

            s = e_betas[k]
            for j in range(0,k):
            # for the summations independent of alpha
                if k>=T_S_li and k<T_E_li:  
                    prob_aa_case7[k] = ( lambda_A[k]*((1-gamma)**2)*L(T,T_S,s,lambda_A))/( lambda_A[k]*(((1-gamma)**2)*L(T,T_S,s,lambda_A)) + lambda_B[k]*(((gamma)**2)*L(T,T_S,s,lambda_B)) ) # corrected 220428
                    prob_bb_case7[k] = ( lambda_B[k]*((gamma)**2)*L(T,T_S,s,lambda_B))/( lambda_A[k]*(((1-gamma)**2)*L(T,T_S,s,lambda_A)) + lambda_B[k]*(((gamma)**2)*L(T,T_S,s,lambda_B)) ) # corrected 220428

    for beta in range(0,D):
        # if beta==D-1:
            # print(f'beta={beta}; delta(T,beta)=')
        for alpha in range(beta+1,D):
            # print(f'beta,alpha={beta,alpha}')
            # if beta==1 and alpha==2:
            t = T[alpha]
            s = e_betas[beta]
            if T_S_li==None or t<T_S and s<T_S: # Case 5 or panmixia

                # Q[beta,alpha] = (1/s)*lambda_A[alpha]*L_s_A[beta,alpha]*J(D,T,lambda_A,alpha,1)*(K(D,T,lambda_A,0,beta,2)*(L_Tse_s_A[beta,beta]**2) + J(D,T,lambda_A,beta,2,upper=s))
                Q[beta,alpha] = (1/s)*lambda_A[alpha]*L_s_A[beta,alpha]*J(D,T,lambda_A,alpha,1)*(K_array[0,beta,0,2]*(L_Tse_s_A[beta,beta]**2) + J(D,T,lambda_A,beta,2,upper=s))
            elif T_S_li!=None and t>=T_S and t<T_E and s<T_S: # Case 6
                prob_a_case6 = (1-gamma)
                prob_b_case6 = gamma
                Q[beta,alpha] = (1/s)*(
                    prob_a_case6*lambda_A[alpha]*(
                        L_s_A[beta,T_S_li]*(1-gamma)*L_A[T_S_li,alpha]*J(D,T,lambda_A,alpha,1)*(K_array[0,beta,0,2]*L_Tse_s_A[beta,beta]**2 + J(D,T,lambda_A,beta,2,upper=s)) 
                        ) + 
                    prob_b_case6*lambda_B[alpha]*(
                        L_s_A[beta,T_S_li]*(gamma)*L_B[T_S_li,alpha]*J(D,T,lambda_B,alpha,1)*(K_array[0,beta,0,2]*L_Tse_s_A[beta,beta]**2 + J(D,T,lambda_A,beta,2,upper=s)) 
                        ))

            elif T_S_li!=None and t>=T_S and t<T_E and s>=T_S and s<T_E: # Case 7
                
                Q[beta,alpha] = (1/s)*(
                    prob_aa_case7[beta]*lambda_A[alpha]*L_s_A[beta,alpha]*J(D,T,lambda_A,alpha,1)*(K_array[0,T_S_li,0,2]*(1-gamma)*L_Tse_s_A[T_S_li,beta]**2 + K_array[T_S_li,beta,0,2]*L_Tse_s_A[beta,beta]**2 + J(D,T,lambda_A,beta,2,upper=s)) +
                    prob_bb_case7[beta]*lambda_B[alpha]*L_s_B[beta,alpha]*J(D,T,lambda_B,alpha,1)*(K_array[0,T_S_li,0,2]*(gamma)*L_Tse_s_B[T_S_li,beta]**2 + K_array[T_S_li,beta,1,2]*L_Tse_s_B[beta,beta]**2 + J(D,T,lambda_B,beta,2,upper=s))
                )

            elif s<T_S and t>=T_E: # Case 8
                prob_a_case8 = (1-gamma)
                prob_b_case8 = gamma                  

                Q[beta,alpha] = (1/s)*lambda_A[alpha]*(
                    prob_a_case8*( (K_array[0,beta,0,2]*L_Tse_s_A[beta,beta]**2 + J(D,T,lambda_A,beta,2,upper=s))*L_s_A[beta,T_S_li]*((1-gamma)*L_A[T_S_li,T_E_li] +gamma)*L_A[T_E_li,alpha]*J(D,T,lambda_A,alpha,1))+
                    prob_b_case8*( (K_array[0,beta,0,2]*L_Tse_s_A[beta,beta]**2 + J(D,T,lambda_A,beta,2,upper=s))*L_s_A[beta,T_S_li]*((gamma)*L_B[T_S_li,T_E_li] +1-gamma)*L_A[T_E_li,alpha]*J(D,T,lambda_A,alpha,1))
                )


            elif t>=T_E and s>=T_S and s<T_E: # Case 9
                
                # prob_aa = (((1-gamma)**2)*L(T_S,s,lambda_A))/(((1-gamma)**2)*L(T_S,s,lambda_A) + ((gamma)**2)*L(T_S,s,lambda_B) ) 
                # prob_bb = (((gamma)**2)*L(T_S,s,lambda_B))/(((1-gamma)**2)*L(T_S,s,lambda_A) + ((gamma)**2)*L(T_S,s,lambda_B) )
                prob_aa = (lambda_A[beta]*( (1-gamma)**2)*L(T,T_S,s,lambda_A)) /( lambda_A[beta]*((1-gamma)**2)*L(T,T_S,s,lambda_A) + lambda_B[beta]*((gamma)**2)*L(T,T_S,s,lambda_B) ) # corrected 220428
                prob_bb = ( lambda_B[beta]*((gamma)**2)*L(T,T_S,s,lambda_B))/( lambda_A[beta]*((1-gamma)**2)*L(T,T_S,s,lambda_A) + lambda_B[beta]*((gamma)**2)*L(T,T_S,s,lambda_B) ) # corrected 220428
                
                Q[beta,alpha] = (1/s)*lambda_A[alpha]*(
                    prob_aa*(
                        L_A[T_E_li,alpha]*J(D,T,lambda_A,alpha,1)*(K_array[0,T_S_li,0,2]*((1-gamma)*(L_Tse_s_A[T_S_li,beta]**2)*L_s_A[beta,T_E_li] + gamma ) + (K_array[T_S_li,beta,0,2]*(L_Tse_s_A[beta,beta]**2) + J(D,T,lambda_A,beta,2,upper=s))*L_s_A[beta,T_E_li])
                    ) +
                    prob_bb*(
                        L_A[T_E_li,alpha]*J(D,T,lambda_A,alpha,1)*(K_array[0,T_S_li,0,2]*((gamma)*(L_Tse_s_B[T_S_li,beta]**2)*L_s_B[beta,T_E_li] + 1-gamma ) + (K_array[T_S_li,beta,1,2]*(L_Tse_s_B[beta,beta]**2) + J(D,T,lambda_B,beta,2,upper=s))*L_s_B[beta,T_E_li])
                    )
                )

            elif t>=T_E and s>=T_E: # Case 10
                Q[beta,alpha] = (1/s)*lambda_A[alpha]*(
                    prob_aa_case10*(K_array[0,T_S_li,0,2]*(L_Tse_s_A[T_E_li,beta]**2)*L_s_A[beta,alpha]*J(D,T,lambda_A,alpha,1)*((1-gamma)*L_A[T_S_li,T_E_li]**2 + gamma)+
                        K_array[T_S_li,T_E_li,0,2]*(L_Tse_s_A[T_E_li,beta]**2)*L_s_A[beta,alpha]*J(D,T,lambda_A,alpha,1) + 
                            (K_array[T_E_li,beta,0,2]*(L_Tse_s_A[beta,beta]**2) + J(D,T,lambda_A,beta,2,upper=s))*L_s_A[beta,alpha]*J(D,T,lambda_A,alpha,1)) +
                    prob_bb_case10*(K_array[0,T_S_li,0,2]*(L_Tse_s_A[T_E_li,beta]**2)*L_s_A[beta,alpha]*J(D,T,lambda_A,alpha,1)*((gamma)*L_B[T_S_li,T_E_li]**2 + 1-gamma)+
                        K_array[T_S_li,T_E_li,1,2]*(L_Tse_s_A[T_E_li,beta]**2)*L_s_A[beta,alpha]*J(D,T,lambda_A,alpha,1) + 
                            (K_array[T_E_li,beta,0,2]*(L_Tse_s_A[beta,beta]**2) + J(D,T,lambda_A,beta,2,upper=s))*L_s_A[beta,alpha]*J(D,T,lambda_A,alpha,1)) + 
                    prob_ab_case10*(K_array[0,T_S_li,0,2]*(L_Tse_s_A[T_E_li,beta]**2)*L_s_A[beta,alpha]*J(D,T,lambda_A,alpha,1)*((1-gamma)*L_A[T_S_li,T_E_li] + gamma*L_B[T_S_li,T_E_li])+
                        0.5*(K_array[T_S_li,T_E_li,0,1] + K_array[T_S_li,T_E_li,1,1])*(L_Tse_s_A[T_E_li,beta]**2)*L_s_A[beta,alpha]*J(D,T,lambda_A,alpha,1) + 
                            (K_array[T_E_li,beta,0,2]*(L_Tse_s_A[beta,beta]**2) + J(D,T,lambda_A,beta,2,upper=s))*L_s_A[beta,alpha]*J(D,T,lambda_A,alpha,1))
                    )
    return Q

# lower diagonal
@njit
def lower_Q(D,T,lambda_A,lambda_B,T_S_li,T_E_li,gamma,e_betas,Q,L_s_A,L_s_B,L_A,L_B,L_Tse_s_A,L_Tse_s_B,K_array):
    # D is int for number of time states
    # T is array of time interval boundaries in coalescent units
    # lambda_A is array of inverse scaled population size parameters for population A
    # lambda_B is array of inverse scaled population size parameters for population B
    # T_S_li is int for index of rejoin time in array T
    # T_E_li is int for index of split time in array T
    # T_S is float for rejoin time in coalescence units
    # T_E is float for split time in coalescence units
    # gamma is float for split fraction
    # e_betas is array of expected coalescence times
    # Q is transitions matrix 
    # L_s_A is array for L(T,) where t1 is s (which does not align with interval boundary) in pop A
    # L_s_B is array for L(T,) where t1 is s (which does not align with interval boundary) in pop B
    # L_A is array for L(T,) where t1 and t2 align with boundaries in pop A
    # L_B is array for L(T,) where t1 and t2 align with boundaries in pop B
    # L_Tse_s_A is array for L(T,) where t2 is s (which does not align with interval boundary) in pop A
    # L_Tse_s_B is array for L(T,) where t2 is s (which does not align with interval boundary) in pop B
    # K_array is an array for the pre computed values of the K function. Iindices correspond to [row,column,pop,factor]
    
    if T_S_li==None:
        T_S = None
        T_E = None
    else:
        T_S = T[T_S_li]
        T_E = T[T_E_li]
    
    # structured pre computations
    if T_S_li!=None:  
        prob_aa_case3 = (((1-gamma)**2)*(L(T,T_S,T_E,lambda_A)))/( ((1-gamma)**2)*(L(T,T_S,T_E,lambda_A)) + ((gamma)**2)*(L(T,T_S,T_E,lambda_B)) + 2*gamma*(1-gamma) )
        prob_bb_case3 = (((gamma)**2)*(L(T,T_S,T_E,lambda_B)))/( ((1-gamma)**2)*(L(T,T_S,T_E,lambda_A)) + ((gamma)**2)*(L(T,T_S,T_E,lambda_B)) + 2*gamma*(1-gamma) )
        prob_ab_case3 = (2*gamma*(1-gamma))/( ((1-gamma)**2)*(L(T,T_S,T_E,lambda_A)) + ((gamma)**2)*(L(T,T_S,T_E,lambda_B)) + 2*gamma*(1-gamma) )
        prob_aa_case4 = (((1-gamma)**2)*(L(T,T_S,T_E,lambda_A)))/( ((1-gamma)**2)*(L(T,T_S,T_E,lambda_A)) + ((gamma)**2)*(L(T,T_S,T_E,lambda_B)) + 2*gamma*(1-gamma) )
        prob_bb_case4 = (((gamma)**2)*(L(T,T_S,T_E,lambda_B)))/( ((1-gamma)**2)*(L(T,T_S,T_E,lambda_A)) + ((gamma)**2)*(L(T,T_S,T_E,lambda_B)) + 2*gamma*(1-gamma) )
        prob_ab_case4 = (2*gamma*(1-gamma))/( ((1-gamma)**2)*(L(T,T_S,T_E,lambda_A)) + ((gamma)**2)*(L(T,T_S,T_E,lambda_B)) + 2*gamma*(1-gamma) )


        prob_a_case2 = np.zeros(D)
        prob_b_case2 = np.zeros(D)
        prob_a_case2_old = np.zeros(D)
        prob_b_case2_old = np.zeros(D)

        for k in range(1,D):
            s = e_betas[k]
            if s>=T_S: # for the L computations in case 2; don't think these save any time
                prob_a_case2[k] = ( lambda_A[k]*((1-gamma)**2)*L(T,T_S,s,lambda_A))/( lambda_A[k]*((1-gamma)**2)*L(T,T_S,s,lambda_A)+lambda_B[k]*(gamma**2)*L(T,T_S,s,lambda_B)) # corrected 220428
                prob_b_case2[k] = ( lambda_B[k]*((gamma)**2)*L(T,T_S,s,lambda_B))/  ( lambda_A[k]*((1-gamma)**2)*L(T,T_S,s,lambda_A)+lambda_B[k]*(gamma**2)*L(T,T_S,s,lambda_B)) # corrected 220428

    for beta in range(0,D): # TODO introduce domain to replace 0 and D, for single slice
        for alpha in range(0,beta):
            # if beta==3 and alpha==2:
            t = T[alpha]
            s = e_betas[beta]
            # print('t is {}, s is {}, T_S is {} and T_E is {}'.format(t,s,T_S,T_E))
            if T_S_li==None or t<T_S: # Case 1 or panmixia

                Q[beta,alpha] = (1/s)*lambda_A[alpha]*(K_array[0,alpha,0,2]*J(D,T,lambda_A,alpha,2) + H(D,T,lambda_A,alpha,2)) 


            elif T_S_li!=None and t>=T_S and t<T_E and s>=T_S and s<T_E: # Case 2

                Q[beta,alpha] = (1/s)*(
                    prob_a_case2[beta]*lambda_A[alpha]*(K_array[0,T_S_li,0,2]*(1-gamma)*(L_A[T_S_li,alpha]**2)*J(D,T,lambda_A,alpha,2) + K_array[T_S_li,alpha,0,2]*J(D,T,lambda_A,alpha,2) + H(D,T,lambda_A,alpha,2))+
                    prob_b_case2[beta]*lambda_B[alpha]*(K_array[0,T_S_li,0,2]*(gamma)*(L_B[T_S_li,alpha]**2)*J(D,T,lambda_B,alpha,2) + K_array[T_S_li,alpha,1,2]*J(D,T,lambda_B,alpha,2) + H(D,T,lambda_B,alpha,2))
                )
                
            elif T_S_li!=None and t>=T_S and t<T_E and s>=T_E: # Case 3  

                Q[beta,alpha] = (1/s)*(
                    prob_aa_case3 * lambda_A[alpha]*(K_array[0,T_S_li,0,2]*(1-gamma)*(L_A[T_S_li,alpha]**2)*J(D,T,lambda_A,alpha,2) + K_array[T_S_li,alpha,0,2]*J(D,T,lambda_A,alpha,2) + H(D,T,lambda_A,alpha,2))+
                    prob_bb_case3 * lambda_B[alpha]*(K_array[0,T_S_li,0,2]*(gamma)*(L_B[T_S_li,alpha]**2)*J(D,T,lambda_B,alpha,2) + K_array[T_S_li,alpha,1,2]*J(D,T,lambda_B,alpha,2) + H(D,T,lambda_B,alpha,2))+
                    prob_ab_case3 * (0.5)*(K_array[0,T_S_li,0,2]*((1-gamma)*(L_A[T_S_li,alpha])*J(D,T,lambda_A,alpha,1)*lambda_A[alpha] + (gamma)*(L_B[T_S_li,alpha])*J(D,T,lambda_B,alpha,1)*lambda_B[alpha] ))
                )

            elif T_S!= None and t>=T_E and s>=T_E: # Case 4

                Q[beta,alpha] = (1/s)*lambda_A[alpha]*(
                    prob_aa_case4*(K_array[0,T_S_li,0,2]*( gamma + (1-gamma)*(L_A[T_S_li,T_E_li]**2) )*(L_A[T_E_li,alpha]**2)*J(D,T,lambda_A,alpha,2) +
                        K_array[T_S_li,T_E_li,0,2]*(L_A[T_E_li,alpha]**2)*J(D,T,lambda_A,alpha,2) + K_array[T_E_li,alpha,0,2]*J(D,T,lambda_A,alpha,2) + H(D,T,lambda_A,alpha,2) ) + 
                    prob_bb_case4*(K_array[0,T_S_li,0,2]*( 1-gamma + (gamma)*(L_B[T_S_li,T_E_li]**2) )*(L_A[T_E_li,alpha]**2)*J(D,T,lambda_A,alpha,2) +
                        K_array[T_S_li,T_E_li,1,2]*(L_A[T_E_li,alpha]**2)*J(D,T,lambda_A,alpha,2) + K_array[T_E_li,alpha,0,2]*J(D,T,lambda_A,alpha,2) + H(D,T,lambda_A,alpha,2) ) +
                    prob_ab_case4*(K_array[0,T_S_li,0,2]*( (1-gamma)*(L_A[T_S_li,T_E_li]) + (gamma)*(L_B[T_S_li,T_E_li]) )*(L_A[T_E_li,alpha]**2)*J(D,T,lambda_A,alpha,2) +
                        0.5*(K_array[T_S_li,T_E_li,0,1]+K_array[T_S_li,T_E_li,1,1])*(L_A[T_E_li,alpha]**2)*J(D,T,lambda_A,alpha,2) + K_array[T_E_li,alpha,0,2]*J(D,T,lambda_A,alpha,2) + H(D,T,lambda_A,alpha,2) ) 
                    )
    return Q

@njit
def J(D,T,lambd,int1,factor,upper=None):
    # D is int for number of time intervals
    # T is array for time interval boundaries in coalescent time
    # lambd is array of lambdas 
    # factor is exponent
    # int1 is int for lower index
    # upper (if given) if float for upper time in coalescent units
    del_ = delta(T,int1) if upper==None else upper - T[int1]
    J = 1/(factor*lambd[int1])*(1-np.exp(-factor*lambd[int1]*del_))
    return J

@njit
def K(D,T,lambd,int1,int2,factor):
    # D is int for number of time intervals
    # T is array for time interval boundaries in coalescent time
    # lambd is array of lambdas
    # factor is exponent
    # int1 is int for lower index
    # # int2 is int for upper index
    # summed_K = Ksummations[int1,int2,pop,factor]
    K=0
    # (1/(2*lambda_A[j]))*(1-np.exp(-2*lambda_A[j]*delta(T,j)))*(L(T,T[j+1],s,lambda_A)**2)
    for j in range(int1,int2):
        K+=J(D,T,lambd,j,factor)*L(T,T[j+1],T[int2],lambd)**factor
    return K

@njit
def H(D,T,lambd,int1,factor,upper=None):
    # D is int for number of time intervals
    # T is array for time interval boundaries in coalescent time
    # lambd is array of lambdas 
    # factor is exponent
    # int1 is int for lower index
    # upper (if given) if float for upper time in coalescent units
    del_ = delta(T,int1) if upper==None else upper - T[int1]
    H = 1/(factor*lambd[int1])*(delta(T,int1) - (1/(factor*lambd[int1]))*(1-np.exp(-factor*lambd[int1]*del_)))
    return H

# def Ksummations_precompute(D,T,lambda_A,lambda_B,T_S_lower_index,T_E_lower_index,gamma,e_betas):
#     summations = np.zeros(shape=(D,D,2,3)) # row, column, pop, factor
#     for j in range(0,D): # col
#         for i in range(j+1,D): # row
#             for pop in range(0,2):
#                 pop_ = [lambda_A,lambda_B][pop]
#                 for f in range(1,3):
#                     if j==0 and i==0:
#                         summations[i,j,pop,f] = J(D,T,pop_,j,f)
#                     elif i==0 and j>0:
#                         summations[i,j,pop,f] = J(D,T,pop_,j,f) + summations[0,j-1,pop_,f]
#                     else:
#                         summations[i,j,pop,f] = summations[0,j,pop,f] - summations[0,i-1,pop,f] # maybe this should be - summations[0,i,pop,f]
#     return summations

@njit
def K_precomputations(D,T,lambda_A,lambda_B,T_S_li,T_E_li):
    # D is int for number of time intervals
    # T is array for time interval boundaries in coalescent time
    # lambda_A and lambda_B is array of lambdas
    # T_S_lower_index is int for index of T_S
    # T_E_lower_index is int for index of T_E

    if T_S_li==None:
        T_S = None
        T_E = None
    else:
        T_S = T[T_S_li]
        T_E = T[T_E_li]

    K_array = np.zeros((D,D,2,3)) # row, column, pop (0 for A 1 for B), factor
    for j in range(0,D):
        K_array[0,j,0,2] = K(D,T,lambda_A,0,j,2) # row 0, column j, population A, factor 2
    if T_S_li!=None: 
        K_array[0,T_S_li,0,2] = K(D,T,lambda_A,0,T_S_li,2) # row 0, column T_S_li, population A, factor 2
        K_array[T_S_li,T_E_li,0,1] = K(D,T,lambda_A,T_S_li,T_E_li,1) # row T_S_li, column T_E_li, population A, factor 1
        K_array[T_S_li,T_E_li,1,1] = K(D,T,lambda_B,T_S_li,T_E_li,1) # row T_S_li, column T_E_li, population B, factor 1

        for k in range(0,D):
            if k>=T_S_li:
                K_array[T_S_li,k,0,2] = K(D,T,lambda_A,T_S_li,k,2) # row T_S_li, column k, population A, factor 2
                K_array[T_S_li,k,1,2] = K(D,T,lambda_B,T_S_li,k,2) # row T_S_li, column k, population B, factor 2
            if k>=T_E_li:
                K_array[T_E_li,k,0,2] = K(D,T,lambda_A,T_E_li,k,2) # row T_E_li, column k, population A, factor 2
    return K_array

class Transition_Matrix:
    def __init__(self,D,spread_1=0.1,spread_2=40,final_T_factor=None,midpoint_transitions=False):

    # def __init__(self,D,lambda_A,lambda_B=None,T_s_index=None,T_e_index=None,gamma=None,spread_1=0.1,spread_2=20):

        """
        inputs:
        -required:
        --D - number of states
        --lambda_A -  set of D elements for relative inverse size of A
        --spread_1 - controlling spread of time intervals, default = 0.1
        --spread_2 - controlling spread of time intervals, default = 20
        
        -unrequired:
        --lambda_B - set of D elements for inverse relative size of B
        -- T_s, T_e, gamma - structured parameters

        other features of self that are calculated:
        -T - the set of D time intervals, defined by time_intervals()
        -e_beta - the "expected time" in interval beta, calculated by e_beta()
        -Q - the transition matrix 

        """
        self.D = D
        self.lambda_A=None
        self.lambda_B=None
        self.T_s_lower_index = None
        self.T_e_lower_index = None
        self.T_s_upper_index = None
        self.T_e_upper_index = None 
        self.T_s=None
        self.T_e=None
        self.gamma=None
        self.spread_1=spread_1
        self.spread_2=spread_2
        self.final_T_factor = final_T_factor
        self.T=time_intervals(self.D,self.spread_1,self.spread_2,self.final_T_factor)
        # self.check=self.check_all() # conditions satisfied, 0 if True # TODO not sure this is the best way to do this TODO in write_tm()
        self.Q_cr = np.zeros(shape=(self.D,self.D)) # transition matrix, conditional on there being a recombination event
        self.Q = np.zeros(shape=(self.D,self.D)) # transition matrix 
        self.midpoint_transitions = midpoint_transitions

        # if self.check==0: # TODO in write_tm
        #     self.e_betas=[self.e_beta(int) for int in range(0,self.D)] # get expected time in each interval
        # if self.T_s is not None: # get index for in T for T_s and T_e # TODO in write_tm()
        #     self.T_s_upper_index, self.T_s_lower_index  = self.index_finder(self.T_s,self.T_s) 
        #     self.T_e_upper_index, self.T_e_lower_index  = self.index_finder(self.T_e,self.T_e)
           



    def write_tm(self,lambda_A,lambda_B,T_S_index,T_E_index,gamma,rho,check=True,exponential=True):
        # must specify structured parameters (lambda_B,T_s_index,T_e_infex,gamma) even if undefined...in which case set them as None
        self.lambda_A = lambda_A
        self.lambda_B = lambda_B
        self.T_S_lower_index = T_S_index
        self.T_E_lower_index = T_E_index
        self.gamma = gamma
        self.rho = rho
        if self.T_S_lower_index!=None:
            self.T_S_upper_index = self.T_S_lower_index + 1
            self.T_S_upper_index = self.T_E_lower_index + 1
            self.T_S = self.T[self.T_S_lower_index]
            self.T_E = self.T[self.T_E_lower_index]
        else:
            self.T_S_upper_index = None
            self.T_S_upper_index = None
            self.T_S = None
            self.T_E = None

        self.e_betas=np.array([e_beta(self.D,self.T,self.lambda_A,self.lambda_B,self.T_S,self.T_E,self.gamma,int,self.midpoint_transitions) for int in range(0,self.D)]) # get expected time in each interval

        if check==True:
            checkparams=check_all(self.D,self.T,self.lambda_A,self.lambda_B,self.T_S,self.T_E,self.gamma)
            if checkparams==1:
                print('There is a problem in the parameters given!')

        # compute values for L function and K      
        L_s_A,L_s_B,L_A,L_B,L_Tse_s_A,L_Tse_s_B = L_precomputations(self.D,self.T,self.lambda_A,self.lambda_B,self.T_S_lower_index,self.T_E_lower_index,self.gamma,self.e_betas)
        K_array = K_precomputations(self.D,self.T,self.lambda_A,self.lambda_B,self.T_S_lower_index,self.T_E_lower_index)
        

        # Ksummations = Ksummations_precompute(self.D,self.T,self.lambda_A,self.lambda_B,self.T_S_lower_index,self.T_E_lower_index,self.gamma,self.e_betas)
        # K_array = K_precomputations(self.D,self.T,self.lambda_A,self.lambda_B,self.T_S_lower_index,self.T_E_lower_index,self.gamma,self.e_betas,Ksummations)               
        # write the transition matrix; both (conditional on recombination events occuring) and regular
        self.Q = upper_Q(self.D,self.T,self.lambda_A,self.lambda_B,self.T_S_lower_index,self.T_E_lower_index,self.gamma,self.e_betas,self.Q,L_s_A,L_s_B,L_A,L_B,L_Tse_s_A,L_Tse_s_B,K_array)
        # self.Q = lower_Q(self.D,self.T,self.lambda_A,self.lambda_B,self.T_S_lower_index,self.T_E_lower_index,self.gamma,self.e_betas,self.Q,L_s_A,L_s_B,L_A,L_B,L_Tse_s_A,L_Tse_s_B,K_array)
        self.Q = lower_Q(self.D,self.T,self.lambda_A,self.lambda_B,self.T_S_lower_index,self.T_E_lower_index,self.gamma,self.e_betas,self.Q,L_s_A,L_s_B,L_A,L_B,L_Tse_s_A,L_Tse_s_B,K_array)


        self.Q[range(self.D),range(self.D)] = 0 # make sure diagonals are 0...if optimising this may not be the case 
        if self.rho!=1:
            self.Q_cr = np.copy(self.Q)
            midpoints = np.array([(self.T[i] + self.T[i+1])/2 for i in range(0,self.D)])
            if exponential==True:
                recomb_probabilities = 1-np.exp(-self.rho*np.array(self.e_betas)) # probability of recombining for each time interval 
            elif exponential==False:
                recomb_probabilities = np.array(self.e_betas)*self.rho # probability of recombining for each time interval 
            self.Q = self.Q*recomb_probabilities[:,np.newaxis]
            diags_cr = [1-np.sum(self.Q_cr[i,:]) for i in range(0,self.D)]
            self.Q_cr[range(self.D),range(self.D)] = diags_cr
            diags = [1-np.sum(self.Q[i,:]) for i in range(0,self.D)]
            self.Q[range(self.D),range(self.D)] = diags
        else: # given that a recomb happens
            diags = [1-np.sum(self.Q[i,:]) for i in range(0,self.D)]
            self.Q[range(self.D),range(self.D)] = diags            

        return self.Q 

# from structured_transition_matrix_211220 import *
# from structured_transition_matrix_210915 import *
# D = 30
# lambda_A = np.ones(D)
# tm_211220 = Transition_Matrix_211220(D=D,spread_1=0.1,spread_2=20) # initialise transition matrix object
# tm_210915 = Transition_Matrix_210915(D=D,spread_1=0.1,spread_2=20) # initialise transition matrix object
# lambda_A = np.ones(D)
# lambda_B = np.ones(D)
# T_s_index = 8
# T_e_index = 22
# gamma = 0.2
# prob_recomb = 1e-08
# N_true = 10000
# N = N_true
# rho = 4*N_true*prob_recomb
# Q_211220 = tm_211220.write_tm(lambda_A=lambda_A,lambda_B=lambda_B,T_s_index=T_s_index,T_e_index=T_e_index,gamma=gamma,check=True,rho=rho) # initialise transition matrix object
# Q_210915 = tm_210915.write_tm(lambda_A=lambda_A,lambda_B=lambda_B,T_s_index=T_s_index,T_e_index=T_e_index,gamma=gamma,check=True,prob_recomb=prob_recomb,N=10000) # initialise transition matrix object
