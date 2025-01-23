#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 16:47:05 2025

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


################################################################################

if __name__ == "__main__":

    L      = 9
    Lrmt   = L-1
    dim    = 2**L
    dimRmt = 2**Lrmt
    mlsgue    = 4./(2*np.pi*dimRmt)
    realizations = 10
    
    eps = 0.03

    t0 = 0 # Initial time value.
    dt = 1e-03#1e-03 # Value of the time step.
    ts = 1000 # Number of time steps (for every scale).
    su = 14#8 # Number of scales spanned.

    # ### Sum_SFF_disc = np.array([])
    # Sum_SFF          = np.zeros(ts*su, dtype=complex)
    # Sum_Part_fct     = np.zeros(ts*su, dtype=complex)
    # Sum_Part_fct_d   = np.zeros(ts*su, dtype=complex)

    # SFF_t            = np.zeros(ts*su, dtype=complex)
    # SFF_d            = np.zeros(ts*su, dtype=complex)
    # SFF_d_d          = np.zeros(ts*su, dtype=complex)
    
    #Load Evals:
    #evals = np.load(os.getcwd() + "/SYK_evals_for_r=%s_eps=%s_L=%s.npy"%(realizations,eps,L))#unscaled eigenvalues
    
    evals = np.load(os.getcwd() + "/SYK_evals_scaled_for_r=%s_eps=%s_L=%s.npy"%(realizations,eps,L))#scaled eigenvalues
    
    length  = ts*su # for predefinition of arrays
    all_SFF  = np.zeros(length, dtype=complex)
    all_U    = np.zeros(length, dtype=complex)
    all_Udag = np.zeros(length, dtype=complex)

    # Computing SFF for cSYk q4
    for r in range(realizations):
        
        evalsc_H = evals[r]
        
        times, SFF, U, Udag = make_sff(evalsc_H, t0, dt, ts, su, 1) 
        all_SFF  += np.array(SFF)
        all_U    += np.array(U)
        all_Udag += np.array(Udag)
        
    # Average the TwoPtVals over the realizations
    avg_SFF  = np.array(all_SFF) / realizations
    avg_U    = np.array(all_U) / realizations
    avg_Udag = np.array(all_Udag) / realizations
    
    avg_SFF_disc = avg_U*avg_Udag
    avg_SFF_conn = (avg_SFF - avg_SFF_disc)/(dimRmt**2)

###Figure out scalings!


    np.save("tVals_L=%s.npy"%(L),times)
    np.save("cSFF_e=%s_L=%s_r=%s.npy"%(eps,L,realizations), avg_SFF_conn)



    #     evals = np.load(os.getcwd() + '/scaled_evals/SCevalsHp_e=%s_L=%s_r=%s.npy'%(eps,L,r))
        
    #     ### Computing the SFF for the perturbed Hamiltonian    with p4
    #     tVals,SFF_tot,Part_fct,Part_fct_d = SFF(evals,t0,dt,ts,su)
    #     SFF_t           = np.array(SFF_tot) # For the total (perturbed) Hamiltonian
    #     Part_fct        = np.array(Part_fct)
    #     Part_fct_d      = np.array(Part_fct_d)

    #     Sum_SFF        +=  SFF_t # For the total (perturbed) Hamiltonian
    #     Sum_Part_fct   +=  Part_fct
    #     Sum_Part_fct_d +=  Part_fct_d

    # #perturbed 4 with final scaling 
    # SFF_t    = Sum_SFF/(1.*realizations)
    # SFF_d    = Sum_Part_fct/(1.*realizations)
    # SFF_d_d  = Sum_Part_fct_d/(1.*realizations)


    # # Spectral Form Factor for the perturbed 4 Hamiltonian
    # SFF_disc  = SFF_d * SFF_d_d
    # SFF       = SFF_t # Normalized
    # SFF_conn  = (SFF - SFF_disc)/dimRmt**2


    # #np.save("sff_H4_e=%s_L=%s.npy"%(epsilon,L), p4SFF_conn)
    # np.save("sff_syk_evenSec_e=%s_L=%s.npy"%(eps,L), SFF_conn)
    # np.save("tVals_syk_evenSec_L=%s.npy"%(L), tVals)





#Test:
import matplotlib.pyplot as plt

plt.plot(times, [1/dimRmt]*len(times), label='full')
plt.plot(times, avg_SFF_conn, label='conn')

plt.legend()
plt.xscale('log')
plt.yscale('log')
plt.show()
