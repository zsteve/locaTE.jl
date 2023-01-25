module locaTE

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
using NMF
using TensorToolbox
using PyCall
using CUDA
using LogExpFunctions
using EvalMetrics
using Base.Threads
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
export get_MI!,
    get_joint_cache, getcoupling_dense, getcoupling_dense_trimmed, getcoupling_sparse
export to_backward_kernel
export estimate_TE, estimate_TE_cu

function __init__()
    return copy!(tl_decomp, pyimport_conda("tensorly.decomposition", "tensorly"))
end

function to_backward_kernel(P)
        π_unif = fill(1/size(P, 1), size(P, 1))'
        (P' .* π_unif)./(π_unif * P)';
end

function estimate_TE(X::Matrix, regulators, targets, P, QT, R; 
        clusters=nothing, 
        discretizer_alg=DiscretizeBayesianBlocks(),
        showprogress=true,
        wclr=false)
    @assert length(regulators) == length(targets) # for now 
    clusters = clusters === nothing ? I(size(X, 1)) : clusters 
    TE = zeros(size(clusters, 2), length(regulators)*length(targets))
    disc = discretizations_bulk(X; alg = discretizer_alg)
    gene_idxs = vcat([[j, i]' for i in regulators for j in targets]...)
    p = showprogress ? Progress(size(clusters, 2)) : nothing 
    @threads for i = 1:size(clusters, 2)
        TE[i, :] = get_MI(
        X,
        compute_coupling(X, i, P, QT, R),
        gene_idxs[:, 1],
        gene_idxs[:, 2];
        disc = disc,
        alg = discretizer_alg,
        )
        if showprogress
            next!(p)
        end
    end
    if wclr
        TE_clr = apply_wclr(mi_all, length(regulators)) # todo
        TE_clr[isnan.(mi_all_clr)] .= 0;
        return TE_clr
    else
        return TE
    end
end

# expects all inputs to reside on CPU
function estimate_TE_cu(X::Matrix, regulators, targets, P, QT, R;
        clusters=nothing, 
        discretizer_alg=DiscretizeBayesianBlocks(),
        showprogress=true,
        wclr=false,
        N_blocks=1)
    @assert length(regulators) == length(targets) # for now 
    clusters = clusters === nothing ? I(size(X, 1)) : clusters 
    disc = discretizations_bulk(X; alg = discretizer_alg)
    disc_max_size = maximum(map(x -> length(x[1]) - 1, disc))
    joint_cache = get_joint_cache(length(regulators) ÷ N_blocks, disc_max_size);
    ids_cu = hcat(map(x -> x[2], disc)...) |> cu;
    # Copy transition matrices and neighbourhood kernel to CUDA device
    P_cu = cu(P)
    QT_cu = cu(QT)
    R_cu = cu(R)
    # Estimate TE using GPU 
    TE = CuArray{Float32}(undef, (size(clusters, 1), length(regulators), length(targets)))
    for i = 1:size(clusters, 1)
        gamma, idx0, idx1 = getcoupling_dense_trimmed(i, P_cu, QT_cu, R_cu)
        for ((N_x, N_y), (offset_x, offset_y)) in getblocks(size(X, 2), N_blocks, N_blocks)
            get_MI!(
                view(TE, i, :, :),
                joint_cache,
                gamma,
                length(regulators),
                ids_cu[idx0, :],
                ids_cu[idx1, :];
                offset_x = offset_x,
                N_x = N_x,
                offset_y = offset_y,
                N_y = N_y,
            )
        end
    end
    # Copy back to CPU
    if wclr
        TE_clr = apply_wclr(Array(reshape(TE, size(clusters, 1), :)), length(regulators)) # todo
        TE_clr[isnan.(TE_clr)] .= 0;
        return TE_clr
    else
        return Array(reshape(TE, size(clusters, 1), :))
    end
end

end # module
