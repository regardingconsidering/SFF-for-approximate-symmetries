#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  6 13:41:02 2025

@author: rahel
"""
import numpy as np
from scipy.linalg import block_diag
import numpy.linalg as la

# =============================================================================

# Building a block-diagonal GUE random matrix of Q-blocks with size N*N:
def makeH_0(N,Q):
    
    n = N

    h_0_list = [None]*Q

    for k in range(0,Q):
            # Define scaling of distribution
            sc = 1./(np.sqrt(2.*n))

            # A, B, C are generated for the constructing the real, imaginary and the diagonal elements of the H_0 Hamiltonian
            a_0 = np.random.normal(scale = sc, size = (n,n))
            b_0 = np.random.normal(scale = sc, size = (n,n))
            c_0 = np.random.normal(scale = 1.0/(np.sqrt(n)), size = (n,1))

            # Define one block of the matrix H_0:
            h_0 = np.zeros((n,n), dtype=complex)

            for i in range(n):
                h_0[i,i] = c_0[i]

            for i in range(n-1):
                for j in range(i+1,n):
                    h_0[i,j] = a_0[i,j] + 1.j*b_0[i,j]
                    h_0[j,i] = a_0[i,j] - 1.j*b_0[i,j]

            h_0_list[k] = h_0

    H_0 = np.zeros((0,0),dtype=complex)
    
    # Using direct product from scipy library to build full Hamiltonian 
    for l in range(0,Q):
        H_0 = block_diag(H_0,h_0_list[l])

    return H_0

# =============================================================================
# Building a nonlocal perturbation matrix V, resp. a NQxNQ hermitian matrix, (with zero diagonal blocks)

def makeV_nonlocal(N,Q):
    n = N*Q
    
    # Define scaling of distribution
    sc = 1./(np.sqrt(2.*N)) # Here we need 'only' an N scaling, since we want to compare the local and nonlocal perturbation with same interaction strengths

    # A, B, C are generated for the constructing the real, imaginary and the diagonal elements of the H_0 Hamiltonian
    a_0 = np.random.normal(scale = sc, size = (n,n))
    b_0 = np.random.normal(scale = sc, size = (n,n))
    c_0 = np.random.normal(scale = 1.0/(np.sqrt(n)), size = (n,1))

    # Define one block of the matrix H_0:
    V_nl = np.zeros((n,n), dtype=complex)

    for i in range(n):  
        V_nl[i,i] = c_0[i]

    for i in range(n-1):
        for j in range(i+1,n):
            V_nl[i,j] = a_0[i,j] + 1.j*b_0[i,j]
            V_nl[j,i] = a_0[i,j] - 1.j*b_0[i,j]
            
    #setting QxQ blocks on diagonal to zero
    # range(0, n, block_size) goes from o to n(exclusive) by a step of 'blocksize=N' 

    for i in range(0, n, N):
        V_nl[i:i+N, i:i+N] = 0
        
    
    return V_nl
    

# =============================================================================

# Building a block-diagonal GUE random matrix of Q-blocks with size N*N:
def makeH_rmt(N,Q):

    n = N*Q

    # Define scaling of distribution
    sc = 1./(np.sqrt(2.*n))

    # A, B, C are generated for the constructing the real, imaginary and the diagonal elements of the H_0 Hamiltonian
    a_0 = np.random.normal(scale = sc, size = (n,n))
    b_0 = np.random.normal(scale = sc, size = (n,n))
    c_0 = np.random.normal(scale = 1.0/(np.sqrt(n)), size = (n,1))

    # Initialise the matrix with 0's
    H_rmt = np.zeros((n,n), dtype=complex)
    
    # Fill diagonal part
    for i in range(n):
        H_rmt[i,i] = c_0[i]
        
    #Fill off-diagonal parts:
    for i in range(n-1):
        for j in range(i+1,n):
            H_rmt[i,j] = a_0[i,j] + 1.j*b_0[i,j]
            H_rmt[j,i] = a_0[i,j] - 1.j*b_0[i,j] # Hermiticity constraint

    return H_rmt


# =============================================================================

if __name__ == "__main__":

    #######################
    # Values to be adapted:
    #######################
    N   = 2**7 
    Q   = 2**2 
    eps = 0.1 
    realizations = 5
    #######################

    dim = Q*N
    mls = 4./(2*np.pi*dim)

    
    H_rmt = makeH_rmt(N,Q)
    evals_H_rmt = la.eigvalsh(H_rmt)
    rmt = np.mean(np.absolute(evals_H_rmt))
    
    # List to store all realizations of eigenvalues
    all_evals    = []
    all_evals_sc = []

    #Computing SFF for block diagonal RMT's with off-diagonal perturbations:
    for r in range(realizations):
        
        # Create H_0 and perturbation V
        H_0   = makeH_0(N,Q)
        V     = makeV_nonlocal(N,Q)
        H     = H_0 + eps*V

        # Compute eigenvalues of H         
        evals_H     = la.eigvalsh(H)
        
        # Append eigenvalues to the list
        all_evals.append(evals_H)
        
        # scaling such that all mls are equal
        #sc_i = np.mean(np.absolute(evals_H_rmt))/np.mean(np.absolute(evals_H))
        sc_f = np.absolute(np.max(evals_H_rmt))/np.absolute(np.max(evals_H))


        #evalsci_H      = evals_H*sc_i/mls
        SCevals_H      = evals_H*sc_f/mls
        
        # Append eigenvalues to the list
        all_evals_sc.append(SCevals_H)
        
    # Convert the list of arrays to a single numpy array (or keep as list)
    all_evals    = np.array(all_evals) # Shape will be (realizations, number_of_eigenvalues_per_realization)
    all_evals_sc = np.array(all_evals_sc)
    
    # Save the entire array in a single .npy file
    np.save("evals_for_r=%s_eps=%s_QN=%s*%s.npy" % (realizations, eps, Q, N), all_evals)
    np.save("evals_scaled_for_r=%s_eps=%s_QN=%s*%s.npy" % (realizations, eps, Q, N), all_evals_sc)


# =============================================================================


