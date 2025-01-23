#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 10:10:15 2025

@author: rahel
"""

import numpy as np
import os


# =============================================================================
# # Spectral Form Factor
def make_sff(evals, t0, dt, ts, su, D):
    
    ###
    tscl  = [dt*2**k for k in range(su)]
    tVals = [t0+(sc-dt)*(1.+ts)+sc*j for sc in tscl for j in range(ts)]
    
    SFF     = np.zeros((ts*su),dtype = 'complex_')
    U       = np.zeros((ts*su), dtype = 'complex_')
    Udagger = np.zeros((ts*su), dtype = 'complex_')

    for i,t in enumerate(tVals):

        U[i] = np.trace(np.diag(np.exp(- 1j * evals * t )))

        Udagger[i] = np.trace(np.diag(np.exp(1j * evals * t )))

        SFF[i] = SFF[i] + U[i] * Udagger[i] / D**2
    
    return tVals,SFF,U,Udagger

#==============================================================================

if __name__ == "__main__":

    #######################
    # Values to be adapted:
    #######################
    N   = 2**8 
    Q   = 2**2 
    eps = 0.1
    realizations = 10
    dim = Q*N
    #######################
    
    t0 = 0
    dt = 1e-04
    ts = 1000
    su = 10 

    #Load Evals:
    evals = np.load(os.getcwd() + "/evals_scaled_for_r=%s_eps=%s_QN=%s*%s.npy"%(realizations,eps,Q,N))
    
    length  = ts*su # for predefinition of arrays
    all_SFF  = np.zeros(length, dtype=complex)
    all_U    = np.zeros(length, dtype=complex)
    all_Udag = np.zeros(length, dtype=complex)

    #Computing SFF for block diagonal RMT's with off-diagonal perturbations:
    for r in range(realizations):


        evalsc_H = evals[r]
        
        times, SFF, U, Udag = make_sff(evalsc_H, t0, dt, ts, su, dim) 
        all_SFF  += np.array(SFF)
        all_U    += np.array(U)
        all_Udag += np.array(Udag)
       
# Average the TwoPtVals over the realizations
avg_SFF  = np.array(all_SFF) / realizations
avg_U    = np.array(all_U) / realizations
avg_Udag = np.array(all_Udag) / realizations
avg_SFF_disc = (avg_U*avg_Udag)/dim**2
avg_SFF_conn = avg_SFF - avg_SFF_disc


np.save("tVals_QN=%s*%s.npy"%(Q,N),times)
np.save("cSFF_e=%s_Q*N=%s*%s_r=%s.npy"%(eps,Q,N,realizations), avg_SFF_conn)


# #Test:
# import matplotlib.pyplot as plt

# plt.plot(times, [1/dim]*len(times))
# plt.plot(times, avg_SFF_conn)
# plt.xscale('log')
# plt.yscale('log')
# plt.show()

