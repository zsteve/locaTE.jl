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
using Plots: findnz

include("inference.jl")
include("util.jl")
include("discretize.jl")
include("opt.jl")

export get_MI_undir, get_MI
export CLR, wCLR, apply_clr
export compute_coupling
export fitsp

end # module
