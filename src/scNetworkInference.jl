module scNetworkInference

using NNlib
using StatsBase
using SparseArrays
using InformationMeasures
using LinearAlgebra
using ProgressMeter
using MultivariateStats
using NearestNeighbors
using Discretizers
using Distributions 
using Plots: findnz
using NMF
using TensorToolbox
using PyCall
using CUDA
using LogExpFunctions
using SparseArrays
using EvalMetrics 
CUDA.allowscalar(false)

const tl_decomp = PyCall.PyNULL()

include("inference.jl")
include("util.jl")
include("discretize.jl")
include("opt.jl")
include("gpu.jl")

export get_MI
export CLR, wCLR, apply_clr, apply_wclr
export compute_coupling
export fitsp
export aupr, prec_rec_rate, ep
export auroc, tp_fp_rate
export fitnmf, fitntf
export get_MI!, get_joint_cache, getcoupling_dense, getcoupling, getcoupling_sparse

function __init__()
    return copy!(tl_decomp, pyimport("tensorly.decomposition"))
end

end # module
