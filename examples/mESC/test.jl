using Pkg
Pkg.activate(".")
using Revise
Pkg.develop(path = "../../")

using NPZ
using NNlib
using OptimalTransport
using StatsBase
using SparseArrays
using LinearAlgebra
using Plots
using LaTeXStrings
using Graphs
using GraphSignals
using NearestNeighbors
using ProgressMeter
using Discretizers
import locaTE as lTE;
using Suppressor
using CSV, DataFrames
using Printf
pyplot()

DATA_DIR = "../HSPC/data"
X = relu.(npzread(joinpath(DATA_DIR, "X.npy")) .- 1e-2)
X_pca = npzread(joinpath(DATA_DIR, "X_pca.npy"))
P = npzread(joinpath(DATA_DIR, "P_kappavelo.npy"))
R = npzread(joinpath(DATA_DIR, "R.npy"))

k = 1
Q = lTE.to_backward_kernel(P)
P_sp = sparse((P^k))
QT_sp = sparse((Q^k)')
R_sp = sparse(R); 

L = lTE.construct_normalized_laplacian(X_pca, 25);

# TE = lTE.estimate_TE(
#     X,
#     1:10,
#     1:10,
#     P_sp,
#     QT_sp,
#     R_sp,
# );

using BenchmarkTools 

i = 1
disc = lTE.discretizations_bulk(X);
targets, regulators = 1:50, 1:50
gene_idxs = vcat([[j, i]' for i in targets for j in regulators]...);
clusters = I(size(P, 1))
clusters_norm = convert(Matrix{eltype(P)}, clusters)
clusters_norm ./= sum(clusters_norm; dims = 1);

coupling=lTE.compute_coupling(X, clusters_norm[:, i], P_sp, QT_sp, R_sp)

"""
import locaTE.get_MI
using InformationMeasures
function get_MI(
    X::AbstractMatrix,
    coupling::AbstractSparseMatrix,
    genes_prev::Vector{Int},
    genes_next::Vector{Int};
    disc = nothing,
    kwargs...,
)
    @assert length(genes_prev) == length(genes_next)
    mi = Vector{eltype(coupling)}(undef, length(genes_prev))
    disc_prev =
        disc === nothing ?
        [discretization(X[:, i]; kwargs...) for i = 1:size(X, 2)] :
        [(x[1], x[2]) for x in disc]
    disc_next =
        disc === nothing ?
        [discretization(X[:, i]; kwargs...) for i = 1:size(X, 2)] :
        [(x[1], x[2]) for x in disc]
    disc_max_size = mapreduce(x -> length(x[1]), max, disc)
    joint_cache = Array{eltype(coupling)}(undef, disc_max_size, disc_max_size, disc_max_size)
    # (I,J,V) representation of coupling 
    I,J,V=findnz(coupling)
    for j = 1:length(genes_prev)
        fill!(joint_cache, 0)
        lTE._discretized_joint_distribution!(
            joint_cache, 
            I, J, V,
            X,
            genes_prev[j],
            genes_next[j],
            disc_prev,
            disc_next;
        )
        mi[j] = InformationMeasures.get_conditional_mutual_information(joint_cache)
    end
    mi
end
"""

# 10s, 5.26gb alloc
# 5s 38mb 
@benchmark get_MI(X, coupling, gene_idxs[:, 1], gene_idxs[:, 2]; disc = disc)

TE = lTE.get_MI(X, lTE.compute_coupling(X, clusters_norm[:, i], P_sp, QT_sp, R_sp), gene_idxs[:, 1], gene_idxs[:, 2]; disc = disc)

# 3ms, 10.42mb
@benchmark lTE.compute_coupling($X, $(clusters_norm[:, i]), $P_sp, $QT_sp, $R_sp)

# ~20us, 38kb
@benchmark lTE.construct_index_maps(coupling)

row_idxs, row_map, col_idxs, col_map = lTE.construct_index_maps(coupling)

disc_prev =
	disc === nothing ?
		[discretization(X[:, i]; kwargs...) for i = 1:size(X, 2)] :
		[(x[1], x[2]) for x in disc];
disc_next =
	disc === nothing ?
		[discretization(X[:, i]; kwargs...) for i = 1:size(X, 2)] :
		[(x[1], x[2]) for x in disc];

j=1

_I, _J, _V = findnz(coupling);

p = zeros(25, 25, 25)

lTE._discretized_joint_distribution!(
    p, 
    _I, _J, _V,
    X,
    regulators[j],
    targets[j],
    disc_prev,
    disc_next;
)

# 4.1ms, 2.14mb
# 200us
# 128us
# 94us
# 93us  (x2500)
function test_fn!(p, _I, _J, _V, X, gene_idxs, disc_prev, disc_next)
        for j = 1:size(gene_idxs, 1)
        fill!(p, 0)
        lTE._discretized_joint_distribution!(
            p, 
            _I, _J, _V,
            X,
            gene_idxs[j, 1],
            gene_idxs[j, 2],
            disc_prev,
            disc_next;
        )
        end
end
@benchmark test_fn!(p, _I, _J, _V, X, gene_idxs, disc_prev, disc_next)

using InformationMeasures
# 8us
@benchmark get_conditional_mutual_information(p)

