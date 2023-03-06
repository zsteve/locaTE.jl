# locaTE

locaTE is a method for context-specific network inference using _local transfer entropy_. 

# High-level functions
```@docs
estimate_TE
estimate_TE_cu
to_backward_kernel
```

# Low-level functions 
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

# Low-level GPU implementation
```@docs
get_MI!
get_joint_cache
getcoupling_dense
getcoupling_dense_trimmed
getcoupling_sparse
```
