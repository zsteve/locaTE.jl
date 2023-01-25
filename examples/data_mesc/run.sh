#!/bin/bash
# Example of using command line helper script directly 
JULIA_NUM_THREADS=32 julia --project=. ../../src/locaTE_cmd.jl --tau 3 --lambda1 25 --lambda2 0.001 --outdir locaTE_output/ --cutoff 0.3 --gpu X.npy X_pca.npy P_statot.npy R.npy 
