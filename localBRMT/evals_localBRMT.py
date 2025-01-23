#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  6 13:41:02 2025

@author: rahel
"""

import numpy as np
from scipy.linalg import block_diag
import numpy.linalg as la


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
# Building a block-diagonal GUE random matrix of Q-blocks with size N*N:
def makeH_rmt(N,Q):

    n = N*Q

            # Define scaling of distribution
    sc = 1./(np.sqrt(2.*n))

            # A, B, C are generated for the constructing the real, imaginary and the diagonal elements of the H_0 Hamiltonian
    a_0 = np.random.normal(scale = sc, size = (n,n))
    b_0 = np.random.normal(scale = sc, size = (n,n))
    c_0 = np.random.normal(scale = 1.0/(np.sqrt(n)), size = (n,1))

    # Define one block of the matrix H_0:
    H_rmt = np.zeros((n,n), dtype=complex)

    for i in range(n):
        H_rmt[i,i] = c_0[i]

    for i in range(n-1):
        for j in range(i+1,n):
            H_rmt[i,j] = a_0[i,j] + 1.j*b_0[i,j]
            H_rmt[j,i] = a_0[i,j] - 1.j*b_0[i,j]

    return H_rmt

# =============================================================================

# Building the off-diagonal perturbation matrix V
def makeV(N,Q):
    n=N
    v_0_list = [None]*(Q)

    for k in range(0,Q-1):
            # Defining Hilbert space dimension of each block
            n = N
            # Define scaling of distribution
            sc = 1./(np.sqrt(2*n)) ### added n-factor

            # A, B are generated for constructing the imaginary elements of the H_0 Hamiltonian
            a_0 = np.random.normal(scale = sc, size = (n,n))
            b_0 = np.random.normal(scale = sc, size = (n,n))

            # Define one block of the matrix H_0:
            v_0 = np.zeros((n,n), dtype=complex)

            for i in range(n):
                for j in range(n):
                    v_0[i,j] = a_0[i,j] + 1.j*b_0[i,j]

            v_0_list[k] = v_0
    v_0_list[-1] = np.zeros((n,n))

    V_0 = np.zeros((0,0),dtype=complex)

    for l in range(0,Q):
        V_0 = block_diag(V_0,v_0_list[l])

    # Then we shift the blocks to off-diagonal and set the diagonal to zero:
    for m in range(1,Q):
        V_0[(m-1)*n:m*n,m*n:(m+1)*n] = V_0[(m-1)*n:m*n,(m-1)*n:m*n]
        V_0[(m-1)*n:m*n,(m-1)*n:m*n] = np.zeros((n,n), dtype=complex)
    
    # The full perturbation is then:
    V = V_0 + V_0.conj().T

    return V

# =============================================================================

if __name__ == "__main__":

    #######################
    # Values to be adapted:
    #######################
    N   = 2**8 
    Q   = 2**2 
    eps = 0.1 
    realizations = 10
    #######################
    
    
    dim = Q*N
    mls = 4./(2*np.pi*dim) 
    
    # Precomputing eigenvalues of an exact RMT in order to use its mean absolute value as scaling for the pertured spectrum. 
    H_rmt       = makeH_rmt(N,Q)
    evals_H_rmt = la.eigvalsh(H_rmt)
    rmt         = np.mean(np.absolute(evals_H_rmt))
    
    # List to store all realizations of eigenvalues
    all_evals    = []
    all_evals_sc = []

    #Computing SFF for block diagonal RMT's with off-diagonal perturbations:
    for r in range(realizations):

        # Create H_0 and perturbation V
        H_0   = makeH_0(N,Q)
        V     = makeV(N,Q)
        H     = H_0 + eps*V


        # Compute eigenvalues of H         
        evals_H     = la.eigvalsh(H)
        
        # Append eigenvalues to the list
        all_evals.append(evals_H)

        
        # Scaling the eigenvalues (to be able to compare spectrum to full RMT)
        #sc_i = np.mean(np.absolute(evals_H_rmt))/np.mean(np.absolute(evals_H))
        sc_f = np.absolute(np.max(evals_H_rmt))/np.absolute(np.max(evals_H))

        #evalsc_i_H      = evals_H*sc_i/mls
        SCevals_H      = evals_H*sc_f/mls
               

        # Append eigenvalues to the list
        all_evals_sc.append(SCevals_H)
        
        
    # Convert the list of arrays to a single numpy array (or keep as list)
    all_evals    = np.array(all_evals) # Shape will be (realizations, number_of_eigenvalues_per_realization)
    all_evals_sc = np.array(all_evals_sc)
    
    
    # Save the entire array in a single .npy file
    np.save("evals_for_r=%s_eps=%s_QN=%s*%s.npy" % (realizations, eps, Q, N), all_evals)
    np.save("evals_scaled_for_r=%s_eps=%s_QN=%s*%s.npy" % (realizations, eps, Q, N), all_evals_sc)

