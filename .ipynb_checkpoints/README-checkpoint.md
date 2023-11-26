# locaTE.jl

Cell-specific network inference using *loca*l *T*ransfer *E*ntropy.

To get started, please see the [documentation](https://zsteve.phatcode.net/locatedocs/).

Code for reproducing the manuscript figures can be found [here](https://github.com/zsteve/locaTE-paper).

## Command line 

Command-line script is available in `src/locaTE_cmd.jl`, which can be called with arguments
```
usage: locaTE_cmd.jl [--tau TAU] [--k_lap K_LAP] [--lambda1 LAMBDA1]
                     [--lambda2 LAMBDA2] [--outdir OUTDIR]
                     [--suffix SUFFIX] [--cutoff CUTOFF] [--gpu]
                     [--maxiter MAXITER] [-h] [X] [X_rep] [P] [R]

positional arguments:
  X                  path to counts matrix, X
  X_rep              path to dim-reduced representation of X
  P                  path to transition matrix
  R                  path to kernel matrix

optional arguments:
  --tau TAU          power for transition matrix (type: Int64,
                     default: 1)
  --k_lap K_LAP      number of neighbours for Laplacian (type: Int64,
                     default: 15)
  --lambda1 LAMBDA1  (type: Float64, default: 5.0)
  --lambda2 LAMBDA2  (type: Float64, default: 0.01)
  --outdir OUTDIR     (default: "./")
  --suffix SUFFIX     (default: "")
  --cutoff CUTOFF    (type: Float64, default: 0.0)
  --gpu
  --maxiter MAXITER  (type: Int64, default: 1000)
  -h, --help         show this help message and exit
```

Example: 
```
JULIA_NUM_THREADS=32 julia locaTE_cmd.jl --tau 1 --lambda1 25 --lambda2 0.001 --outdir locaTE_output/ --cutoff 0.3 X.npy X_pca.npy P_velo_dot.npy R.npy 
```

## Examples 

For further examples, consult the `examples/` directory. 

## Further information

See the [preprint](https://www.biorxiv.org/content/10.1101/2023.01.08.523176v1).

Zhang, S.Y. and Stumpf, M.P., 2023. 
Dynamical information enables inference of gene regulation at single-cell scale. bioRxiv, pp.2023-01.

