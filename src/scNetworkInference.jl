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

include("inference.jl")
include("util.jl")
include("discretize.jl")
include("opt.jl")

export get_MI_undir, get_MI
export CLR, wCLR, apply_clr, apply_wclr
export compute_coupling
export fitsp
export aupr, prec_rec_rate 

end # module
