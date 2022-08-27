using Pkg
Pkg.develop(path="/home/syz/sshfs/scNetworkInference.jl/")

using Revise
using scNetworkInference
using OptimalTransport
using NPZ
using StatsBase
using SparseArrays
using ProgressMeter
import scNetworkInference as scN
using Plots
using NearestNeighbors
using Graphs
using GraphSignals
using Printf
using Base.Threads
using LinearAlgebra
using DataFrames
using CSV
using Distances
Nq = 500

DATA_PATH="/home/syz/sshfs/SCODE/"
cd(DATA_PATH)

X = npzread("X.npy")
genes = Array(CSV.read("genes.txt", DataFrame)[:, 2])
X_pca = npzread("X_pca.npy")
X_umap = npzread("X_umap.npy")
P = npzread("P_statot.npy")
C = npzread("C.npy");
# C = pairwise(SqEuclidean(), X_pca')
dpt = npzread("dpt.npy");

# select genes
id = 1:50 
X = X[:, id]
genes = genes[id];

R = quadreg(ones(size(X, 1)), ones(size(X, 1)), C, 10*mean(C));
gene_idxs = vcat([[j, i]' for i = 1:size(X, 2) for j = 1:size(X, 2)]...);

k=1
P_sp = sparse(P^k)
π_unif = fill(1/size(P, 1), size(P, 1))'
Q = (P' .* π_unif)./(π_unif * P)';
QT_sp = sparse((Q^k)')
R_sp = sparse(R);

p0 = R[sortperm(dpt)[250], :]
plot(scatter(X_pca[:, 1], X_pca[:, 2]; marker_z = p0), 
     scatter(X_pca[:, 1], X_pca[:, 2]; marker_z = P_sp'*p0 - p0, color = :bwr, clim = (-0.01, 0.01)))

# construct kNN and Laplacian
kdtree = KDTree(X_pca')
idxs, dists = knn(kdtree, X_pca', 25);
A = spzeros(size(X_pca, 1), size(X_pca, 1));
for (i, j) in enumerate(idxs)
    A[i, j] .= 1.0
end
L = sparse(normalized_laplacian(max.(A, A'), Float64));

coupling = compute_coupling(X, 1, P_sp, QT_sp, R_sp)
using BenchmarkTools
@benchmark get_MI($X, $X, $coupling, $(gene_idxs[:, 1]), $(gene_idxs[:, 2]))

get_MI(X, X, coupling, gene_idxs[:, 1], gene_idxs[:, 2])
