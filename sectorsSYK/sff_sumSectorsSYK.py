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
    dt = 1e-03 # Value of the time step.
    ts = 1000 # Number of time steps (for every scale).
    su = 14 # Number of scales spanned.

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


    np.save("tVals_L=%s.npy"%(L),times)
    np.save("cSFF_e=%s_L=%s_r=%s.npy"%(eps,L,realizations), avg_SFF_conn)



##Test:
#import matplotlib.pyplot as plt

#plt.plot(times, [1/dimRmt]*len(times), label='full')
#plt.plot(times, avg_SFF_conn, label='conn')
#plt.legend()
#plt.xscale('log')
#plt.yscale('log')
#plt.show()
