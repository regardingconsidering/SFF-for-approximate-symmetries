#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 16:28:09 2025

@author: rahel
"""

import numpy as np
from scipy.sparse import lil_matrix
import numpy.linalg as la
from scipy.special import binom


# SYK STUFF
# Some auxiliary functions using bitwise logical operators, useful for constructing the Hamiltonian:

def btest(bs, p): # Tests if there is a 1 at position p.
    return ((bs>>p)&1)

def bflip(bs, p): # Flips the bit at position p.
    return bs ^ (1<<p)

def popcnt(x): # Counts the number of "1" in the binary number "x" (which represents an element in the Fock space basis).
    return bin(x).count('1')

def fparity(st,a,b): # Provides the sign factor required due to the fermionic nature of the Fock space.
    assert a<b
    # Count the number of 1s from a (inclusive!) to b (exclusive!):
    return (-1)**popcnt(st & ( ((1<<b)-1) - ((1<<a)-1) ))

def fparitytwo(st,a,b): # Same sign factor, without the assertion "a<b".
    #assert a<b
    # Count the number of 1s from a (inclusive!) to b (exclusive!):
    return (-1)**popcnt(st & ( ((1<<b)-1) - ((1<<a)-1) ))

def sign (st,i): # Sign that c and c_dagger at site i must take:
      # Count the number of 1s from 0 (inclusive!) to i (exclusive!):
    return (-1)**popcnt(st & ( ((1<<i)-1) - ((1<<0)-1)))

################################################################################
################################################################################

def makeH_SYK_q4_sum_sectors(L):
    J = 1
    dim = 2**L
    #N = L//2

    # Real and imaginary parts of J_{ij;kl}:
    ReJijkl = np.random.normal(scale=(np.sqrt(0.5*float(J**2*6)/(L**3))),size=(L,L,L,L))
    ImJijkl = np.random.normal(scale=(np.sqrt(0.5*float(J**2*6)/(L**3))),size=(L,L,L,L))

    # I'm using the lil_matrix sparse format because it allows fast matrix generation.
    # This initializes the Hamiltonian as a nxn matrix of zeroes:
            
    H = lil_matrix((dim,dim),dtype=complex) # here we still want to populate the elements in the overall Hamiltonian

    
    # start loop over N_sectors: 
    # We're keeping L odd and are summing only over even number of particle number sectors (-1)^F = even     
    for n in range(0, L, 2): 
            
        size_sec = int(binom(L, n)) # The dimension of the nth sector   
    
        # Let's enumerate the basis in a sector of fixed particle number:
        # basis is a list that gives the basis state for an index in the basis.
        
        basis = []
        #basis_map = {}
        for st in range(0,2**L):
            if popcnt(st) == n:
                #basis_map[st] = len(basis)
                basis.append(st)
        assert len(basis) == size_sec



       #These loops sum over all off-diagonal elements, that is over elements for which i<j and k<l, and at the same time the ordered pairs satisfy (i,j)\leq(k,l).
        for i in range(0,L-1):
         for j in range(i+1,L):
          for k in range(i,L-1):
           if k > i:
            m = k+1
           else:
            m = max(i+j-k,k+1)
           for l in range(m,L):
            for st in basis:
                # Test whether basis state st is annihilated by the operator.
                # Only non-zero matrix elements of H computed.
                if btest(st,k) and btest(st,l) and (((not btest(st,i)) and (not btest(st,j))) or ((i == k) and (j == l)) or (i == k and (not btest(st,j))) or (j == l and (not btest(st,i))) or (j==k and(not btest(st,i)))):
                    # Add a fermion at position i and j, remove one at position k and l (accomplished by ^ operator)
                    # This is the action of this part of H on basis state st. 
                    # 'sign' accounts for anticommutation of creation operators in the Hamiltonian with those present in the state itself.

                    stp = st ^ ((1<<i)^(1<<j)^(1<<k)^(1<<l)) # The state resulting after acting with this part of the Hamiltonian (i.e. this set of indices) on the state 'st'.
                    sign = fparity(st,i,j)*fparity(st,k,l)*(-1) #### There is possibly an error of a factor of (-1) here but it might not matter because it only sets the overall sign of the Hamiltonian. I have corrected that using the additional factor of (-1).

                    if (stp == st):

                        H[stp,st] += 1.*ReJijkl[i,j,k,l]*sign/np.sqrt(0.5) # As the diagonal coupling strength should still have variance 6 J^2 / L^3, even though it has no imaginary part 

                    elif (stp != st):

                        H[stp,st] += ReJijkl[i,j,k,l]*sign + ImJijkl[i,j,k,l]*1j*sign
                        H[st,stp] += ReJijkl[i,j,k,l]*sign - ImJijkl[i,j,k,l]*1j*sign # The Hamiltonian needs to be hermitian.
   
    


    return H


################################################################################
def makeH_SYK_pert_even(L): # This creates a perturbation that consists of 1 creation operator and 3 annihilation operators plus c.c.
    J = 1
    dim = 2**L
    
    # Real and imaginary parts of J_{ij;kl}:
    ReJijkl = np.random.normal(scale=(np.sqrt(0.5*float(J**2*6)/(L**3))),size=(L,L,L,L))
    ImJijkl = np.random.normal(scale=(np.sqrt(0.5*float(J**2*6)/(L**3))),size=(L,L,L,L))

    # I'm using the lil_matrix sparse format because it allows fast matrix generation.
    # This initializes the Hamiltonian as a nxn matrix of zeroes:
    Hp = lil_matrix((dim,dim),dtype=complex)

    for n in range(0, L, 2): 
            
        size_sec = int(binom(L, n)) # The dimension of the nth sector   
    
        # Let's enumerate the basis in a sector of fixed particle number:
        # basis is a list that gives the basis state for an index in the basis.
        
        basis = []
        #basis_map = {}
        for st in range(0,2**L):
            if popcnt(st) == n:
                #basis_map[st] = len(basis)
                basis.append(st)
        assert len(basis) == size_sec

        

       #These loops sum over all elements in the sum of the perturbation, that is over elements for which i can take any value and j<k<l.
        for i in range(0,L):
         for j in range(0,L-2):
          for k in range(j+1,L-1):
           for l in range(k+1,L):
            for st in basis:
                # Test whether basis state st is annihilated by the operator.
                # Only non-zero matrix elements of H computed.
                if btest(st,j) and btest(st,k) and btest(st,l) and ((not btest(st,i)) or (i == j) or (i == k) or (i == l)):
                    # Add a fermion at position i, remove one at position j and k and l (accomplished by ^ operator)
                    # This is the action of this part of H on basis state st. 
                    # 'sign' accounts for anticommutation of creation operators in the Hamiltonian with those present in the state itself.

                    stp = st ^ ((1<<i)^(1<<j)^(1<<k)^(1<<l)) # The state resulting after acting with this part of the Hamiltonian (i.e. this set of indices) on the state 'st'.
                    if (i>=l): # The sign computations here follow the latter order: c_i^\dagger c_j c_k c_l = - c_i^\dagger c_l c_k c_j
                        sign = (fparity(st,j,k)*fparitytwo(st,l,i)) #modified
                    else:
                        sign = fparity(st,j,k)*fparity((st^((1<<j)^(1<<k))),i,l) # I modified the argument for the second fparity to remove the 1s at the location of j and k to avoid the possibilities of when i might be smaller or bigger than k and/or j.

                    Hp[stp,st] += ReJijkl[i,j,k,l]*sign + ImJijkl[i,j,k,l]*1j*sign
                    Hp[st,stp] += ReJijkl[i,j,k,l]*sign - ImJijkl[i,j,k,l]*1j*sign # The Hamiltonian needs to be hermitian.

    
    return Hp

################################################################################
def makeH_RMT(L):
    # Defining Hilbert space dimension of H_RMT
    n = 2**L #N*Q
    
    # Define scaling of distribution
    sc = 1./np.sqrt(2.*n) #Convention such that dos from -2 to +2

    # A, B, C are generated for the constructing the real, imaginary and the diagonal elements of the H_RMT Hamiltonian
    a_0 = np.random.normal(scale = sc, size = (n,n))
    b_0 = np.random.normal(scale = sc, size = (n,n))

    #c_0 = np.random.normal(scale = 1.0/np.sqrt(n), size = (n))
    c_0 = np.random.normal(scale = np.sqrt(2)*sc, size = (n))
    
    # Define one block of the matrix H_0:
    H_RMT = np.zeros((n,n), dtype=complex)

    for i in range(n):
            H_RMT[i,i] = c_0[i]

    for i in range(n-1):
        for j in range(i+1,n):
            H_RMT[i,j] = a_0[i,j] + 1.j*b_0[i,j]
            H_RMT[j,i] = a_0[i,j] - 1.j*b_0[i,j]


    return H_RMT

################################################################################
################################################################################
if __name__ == "__main__":

    L = 9#13 # to compare with Q=16 N=512
    
    Lrmt   = L-1
    n      = 2**L
    dim    = 2**L
    dimRmt = 2**Lrmt
    mlsgue = 4./(2*np.pi*dimRmt)
    
    eps = 0.03
    
    realizations = 10
    
    Hgue   = makeH_RMT(Lrmt)
    evalsHgue  = la.eigvalsh(Hgue)

    # List to store all realizations of eigenvalues
    all_evals    = []
    all_evals_sc = []

    for r in range(realizations):
        
        #Hgue   = makeH_RMT(Lrmt)
        Heven  = makeH_SYK_q4_sum_sectors(L).toarray()
        V      = makeH_SYK_pert_even(L).toarray()
        
        Hp     = Heven + eps*V        

        #evalsHgue  = la.eigvalsh(Hgue) 
        evalsHp    = la.eigvalsh(Hp)
        
        # Append eigenvalues to the list
        all_evals.append(evalsHp)
        
        #scaling with respect to GUE (eps=1) SFF
        sc_p    = np.absolute(np.max(evalsHgue)/np.max(evalsHp))
        
        #old, just to test:###
        #sc_p    = np.mean(np.absolute(evalsHgue))/np.mean(np.absolute(evalsHp))
        
        ####
        
        SCevalsHp    = evalsHp*sc_p/mlsgue
  
        # Append eigenvalues to the list
        all_evals_sc.append(SCevalsHp)

        
    # Convert the list of arrays to a single numpy array (or keep as list)
    all_evals    = np.array(all_evals) # Shape will be (realizations, number_of_eigenvalues_per_realization)
    all_evals_sc = np.array(all_evals_sc)


    # Save the entire array in a single .npy file
    np.save("SYK_evals_for_r=%s_eps=%s_L=%s.npy" % (realizations, eps, L), all_evals)
    np.save("SYK_evals_scaled_for_r=%s_eps=%s_L=%s.npy" % (realizations, eps, L), all_evals_sc)



