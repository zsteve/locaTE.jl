# locaTE

locaTE is a method for context-specific network inference using _local transfer entropy_. 

# Basic functionality
```@docs
get_MI
CLR
wCLR
apply_clr
apply_wclr
compute_coupling
```

# Denoising and factor analysis
```@docs
fitsp
fitnmf
fitntf
```

# Evaluation
```@docs
aupr
prec_rec_rate
ep
auroc
tp_fp_rate
```

# GPU implementation
```@docs
get_MI!
get_joint_cache
getcoupling
getcoupling_dense
getcoupling_sparse
```