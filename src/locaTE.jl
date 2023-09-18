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
using Graphs
using GraphSignals
using NearestNeighbors

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
export construct_normalized_laplacian
export estimate_TE, estimate_TE_cu
export cdf_norm, apply_cdf_norm, pmean

function __init__()
    return copy!(tl_decomp, pyimport_conda("tensorly.decomposition", "tensorly"))
end

"""
    to_backward_kernel(P::AbstractArray)

Compute backward kernel `QT` from a forward transition kernel `P` using the transpose method. 

"""
function to_backward_kernel(P::AbstractArray)
    π_unif = fill(1 / size(P, 1), size(P, 1))'
    (P' .* π_unif) ./ (π_unif * P)'
end

"""
    construct_normalized_laplacian(X_rep, k)

Construct k-NN graph and normalized, symmetric Laplacian matrix from dimensionality-reduced representation `X_rep`

"""
function construct_normalized_laplacian(X_rep::AbstractArray, k::Int)
    kdtree = KDTree(X_rep')
    idxs, dists = knn(kdtree, X_rep', k);
    A = spzeros(eltype(X_rep), size(X_rep, 1), size(X_rep, 1));
    for (i, j) in enumerate(idxs)
        A[i, j] .= 1
    end
    L = sparse(normalized_laplacian(max.(A, A')));
end

"""
    estimate_TE(
        X::AbstractMatrix,
        regulators,
        targets,
        P::AbstractMatrix,
        QT::AbstractMatrix,
        R::AbstractMatrix;
        clusters = nothing,
        discretizer_alg::DiscretizationAlgorithm = DiscretizeBayesianBlocks(),
        showprogress::Bool = true,
        wclr::Bool = false,
    )

High-level function for estimating local transfer entropy from cell-by-gene expression matrix `X`, with forward transition kernel `P`, 
backward transition kernel `QT` (e.g. calculated from `P` using `to_backward_kernel`), and neighbourhood kernel `R`. 
A subset of regulators and targets can be passed in as index vectors `regulators` and `targets` respectively.
If one seeks to use metacells, a (sparse) Boolean matrix `clusters` can be passed of dimensions `cells × metacells`, encoding the cell-metacell memberships.
A custom discretization algorithm can be passed using `discretizer_alg` (see the documentation of [Discretizers.jl](https://github.com/sisl/Discretizers.jl) for further details)
A progress bar is shown optionally depending on `showprogress`: this is enabled by default.
If `wclr` is set to `true`, a matrix of filtered TE scores is returned in place of raw TE scores: this is disabled by default. 

"""
function estimate_TE(
    X::AbstractMatrix,
    regulators,
    targets,
    P::AbstractMatrix,
    QT::AbstractMatrix,
    R::AbstractMatrix;
    clusters = nothing,
    discretizer_alg::DiscretizationAlgorithm = DiscretizeBayesianBlocks(),
    showprogress::Bool = true,
    wclr::Bool = false,
)
    clusters = clusters === nothing ? I(size(X, 1)) : clusters
    TE = zeros(eltype(X), size(clusters, 2), length(regulators) * length(targets))
    disc = discretizations_bulk(X; alg = discretizer_alg)
    gene_idxs = vcat([[j, i]' for i in targets for j in regulators]...)
    p = showprogress ? Progress(size(clusters, 2)) : nothing
    clusters_norm = convert(Matrix{eltype(P)}, clusters)
    clusters_norm ./= sum(clusters_norm; dims = 1)
    @threads for i = 1:size(clusters, 2)
        TE[i, :] = get_MI(
            X,
            compute_coupling(X, clusters_norm[:, i], P, QT, R),
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
        TE_clr = apply_wclr(TE, length(regulators), length(targets)) 
        TE_clr[isnan.(TE_clr)] .= 0
        return TE_clr
    else
        return TE
    end
end

"""
    estimate_TE_cu(
        X::AbstractMatrix,
        regulators,
        targets,
        P::AbstractMatrix,
        QT::AbstractMatrix,
        R::AbstractMatrix;
        clusters = nothing,
        discretizer_alg::DiscretizationAlgorithm = DiscretizeBayesianBlocks(),
        showprogress::Bool = true,
        wclr::Bool = false,
        N_blocks::Int = 1,
        mode = :dense 
    )

High-level function for estimating local transfer entropy, utilising GPU acceleration. 
The usage for this function is identical to `estimate_TE`, except the TE estimation step is done using a CUDA kernel.
The number of CUDA blocks to be used can be passed as `N_blocks`; by default this is taken to be `1`.
Two modes are available, `mode = :dense` in which a dense representation of a submatrix of the coupling is used, and 
`mode = :sparse` in which a truly sparse representation of the coupling is used. 

"""
function estimate_TE_cu(
    X::AbstractMatrix,
    regulators,
    targets,
    P::AbstractMatrix,
    QT::AbstractMatrix,
    R::AbstractMatrix;
    clusters = nothing,
    disc = nothing,
    discretizer_alg::DiscretizationAlgorithm = DiscretizeBayesianBlocks(),
    showprogress::Bool = true,
    wclr::Bool = false,
    N_blocks::Int = 1,
    mode = :dense
)
    clusters = clusters === nothing ? I(size(X, 1)) : clusters
    p = showprogress ? Progress(size(clusters, 2)) : nothing
    disc = disc === nothing ? discretizations_bulk(X; alg = discretizer_alg) : disc
    disc_max_size = maximum(map(x -> length(x[1]) - 1, disc))
    joint_cache = get_joint_cache(length(regulators) ÷ N_blocks, length(targets) ÷ N_blocks, disc_max_size)
    ids_cu = hcat(map(x -> x[2], disc)...) |> cu
    # Copy transition matrices and neighbourhood kernel to CUDA device
    P_cu = cu(P)
    QT_cu = cu(QT)
    R_cu = cu(R)
    clusters_norm = convert(Matrix{eltype(P_cu)}, clusters)
    clusters_norm ./= sum(clusters_norm; dims = 1)
    clusters_norm_cu = cu(clusters_norm)
    # Estimate TE using GPU 
    TE = CuArray{eltype(joint_cache)}(undef, (size(clusters, 2), length(regulators), length(targets)))
    for i = 1:size(clusters, 2)
        if mode == :dense
            gamma, idx0, idx1 = getcoupling_dense_trimmed(clusters_norm_cu[:, i], P_cu, QT_cu, R_cu)
            for ((N_x, N_y), (offset_x, offset_y)) in getblocks(length(regulators), length(targets), N_blocks, N_blocks)
                get_MI!(
                    view(TE, i, :, :),
                    joint_cache,
                    gamma,
                    ids_cu[idx0, :],
                    ids_cu[idx1, :],
                    regulators,
                    targets;
                    offset_x = offset_x,
                    N_x = N_x,
                    offset_y = offset_y,
                    N_y = N_y,
                )
            end
        elseif mode == :sparse
            I, J, V = getcoupling_sparse(clusters_norm[:, i], P, QT, R)
            for ((N_x, N_y), (offset_x, offset_y)) in getblocks(length(regulators), length(targets), N_blocks, N_blocks)
                get_MI!(
                    view(TE, i, :, :),
                    joint_cache,
                    cu(I), cu(J), cu(Array(V)), 
                    ids_cu,
                    regulators,
                    targets;
                    offset_x = offset_x,
                    N_x = N_x,
                    offset_y = offset_y,
                    N_y = N_y,
                )
            end
        end
        if showprogress
            next!(p)
        end
    end
    # Copy back to CPU
    if wclr
        TE_clr = apply_wclr(Array(reshape(TE, size(clusters, 2), :)), length(regulators), length(targets)) 
        TE_clr[isnan.(TE_clr)] .= 0
        return TE_clr
    else
        return Array(reshape(TE, size(clusters, 2), :))
    end
end

end # module
