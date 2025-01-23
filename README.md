# SFF-for-approximate-symmetries
This code allows the computation of eigenvalues and spectral form factor (SFF) of matrices with approximate symmetries. These codes were used to produce the numerical results in section 3 of the paper "Hilbert Space Diffusion in Systems with Approximate Symmetries", arxiv:2405.19260. 

  **localBRMT**: Considers a toy model built as diagonal matrix with *Q* sectors of size *N^2* sampled from a GUE (referred to as BRMT=block                  Random Matrix Theory). Interactions between nearest neighbouring sectors, hence the name *local* BRMT. 

  **nonlocalBRMT**: Considers a toy model built as diagonal matrix with *Q* sectors of size *N^2* sampled from a GUE. Now interactions                           between any sectors are allowed, hence the name *nonlocal* BRMT. 

  **sectorsSYK**: Considers a physical model, the complex Sachdev-Ye-Kitaev (SYK) model with weak explicit breaking of the U(1) symmetry.                      following a 'local exploration scheme' of sectors.  


Each folder contains a file *evals_xxx.py* building the specific model and computing its eigenvalues, saved in a *.npy* file. And a *sff_xxx.py* file loading the eigenvalues as input data and evaluating the spectral form factor, saving it again in a *.npy* file. 
