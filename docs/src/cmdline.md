# Command line utility  

A script `src/locaTE_cmd.jl` that can be run from command line is included, 
which expects inputs to be passed as `.npy` or `.csv` files. 

```
usage: locaTE_cmd.jl [--tau TAU] [--k_lap K_LAP] [--lambda1 LAMBDA1]
                     [--lambda2 LAMBDA2] [--outdir OUTDIR]
                     [--suffix SUFFIX] [--cutoff CUTOFF] [--gpu]
                     [--maxiter MAXITER] [-h] X X_rep P R

locaTE-cmd: utility for running locaTE workflow from the command-line.
Input matrices can be supplied as .npy (binary) or .csv (text).

positional arguments:
  X                  Path to counts matrix X
  X_rep              Path to dimensionality-reduced representation of
                     X. From this, the kNN graph will be constructed.
  P                  Path to transition matrix P encoding dynamics.
  R                  Path to kernel matrix R encoding neighbourhood
                     information.

optional arguments:
  --tau TAU          Power for transition matrix. (type: Int64,
                     default: 1)
  --k_lap K_LAP      Number of neighbours for Laplacian. (type: Int64,
                     default: 15)
  --lambda1 LAMBDA1  lambda1 (λ1), strength of Laplacian
                     regularization. (type: Float64, default: 5.0)
  --lambda2 LAMBDA2  lambda2 (λ2), strength of Lasso regularization.
                     (type: Float64, default: 0.01)
  --outdir OUTDIR    Output directory (default: "./")
  --suffix SUFFIX    Suffix to append to output files. (default: "")
  --cutoff CUTOFF    Cutoff below which expression values will be set
                     to zero. Can be helpful for expression value
                     binning in some datasets with artifactually small
                     counts. (type: Float64, default: 0.0)
  --gpu              GPU acceleration, recommended for large datasets
                     when available.
  --maxiter MAXITER  Maximum iterations for denoising regression.
                     (type: Int64, default: 1000)
  -h, --help         show this help message and exit
```
