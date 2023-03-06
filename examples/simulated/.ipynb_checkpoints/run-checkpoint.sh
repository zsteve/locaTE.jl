#!/bin/bash
# Example of using command line helper script directly 
JULIA_NUM_THREADS=32 julia --project=. ../../src/locaTE_cmd.jl --tau 1 --lambda1 25 --lambda2 0.001 --outdir locaTE_output/ --cutoff 0.3 data/X.npy data/X_pca.npy data/P_velo_dot.npy data/R.npy 
