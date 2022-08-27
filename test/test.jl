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
using LinearAlgebra

DIR="/home/syz/sshfs/BoolODE/Synthetic/dyn-BF/dyn-BF-500-1"

X = npzread("$DIR/X.npy")
X_pca = npzread("$DIR/X_pca.npy")
X_tsne = npzread("$DIR/X_umap.npy")
# P = npzread("$DIR/P_velo_corr.npy")
P = npzread("$DIR/P_statot.npy")
C = npzread("$DIR/C.npy")
dpt = npzread("$DIR/dpt.npy")

R = quadreg(ones(size(X, 1)), ones(size(X, 1)), C, 2.5*median(C))

gene_idxs = vcat([[j, i]' for i = 1:size(X, 2) for j = 1:size(X, 2)]...);

k = 3
P_sp = sparse(P^k)
π_unif = fill(1/size(P, 1), size(P, 1))'
Q = (P' .* π_unif)./(π_unif * P)';
QT_sp = sparse((Q^k)')
R_sp = sparse(R)

# construct kNN and Laplacian
kdtree = KDTree(X_pca')
idxs, dists = knn(kdtree, X_pca', 25);
A = spzeros(size(X_pca, 1), size(X_pca, 1));
for (i, j) in enumerate(idxs)
    A[i, j] .= 1.0
end
L = sparse(normalized_laplacian(max.(A, A'), Float64));

# directed inference
@info "Directed inference"
mi_all = zeros(size(X, 1), size(X, 2)^2);
p = Progress(size(X, 1))
@info "Computing RDI scores"
for i = 1:size(X, 1)
    mi_all[i, :] = get_MI(X, X, compute_coupling(X, i, P_sp, QT_sp, R_sp), gene_idxs[:, 1], gene_idxs[:, 2])
    next!(p)
end

w = vec(sum(mi_all; dims = 2))
w ./= mean(w)

scatter(X_tsne[:, 1], X_tsne[:, 2]; marker_z = w)

@info "Applying CLR"
mi_all_clr = apply_wclr(mi_all, size(X, 2))
@info "Denoising"

G = fitsp(mi_all_clr, L; λ1 = 5, λ2 = 1e-3, maxiter = 250)

plt=plot(heatmap(mi_all[sortperm(dpt), :]; title = "MI"), 
    heatmap(mi_all_clr[sortperm(dpt), :]; title = "MI+CLR"), 
         heatmap(G[sortperm(dpt), :]; title = "denoised"), layout = (1, 3), size = (750, 250))
savefig(plt, "scGRNs_directed.png")

# undirected inference
@info "Undirected inference"
mi_all = zeros(size(X, 1), size(X, 2)^2);

p = Progress(size(X, 1))
@info "Computing RDI scores"
for i = 1:size(X, 1)
    mi_all[i, :] = get_MI_undir(X, compute_coupling(X, i, R_sp), gene_idxs[:, 1], gene_idxs[:, 2])
    next!(p)
end
@info "Applying CLR"
mi_all_clr = apply_clr(mi_all, size(X, 2))
@info "Denoising"
G = fitsp(mi_all_clr, L; λ1 = 10, λ2 = 0.5, maxiter = 250)

plt=plot(heatmap(mi_all[sortperm(dpt), :]; title = "MI"), 
    heatmap(mi_all_clr[sortperm(dpt), :]; title = "MI+CLR"), 
         heatmap(G[sortperm(dpt), :]; title = "denoised"), layout = (1, 3), size = (750, 250))
savefig(plt, "scGRNs_undirected.png")


###
x = R[sortperm(dpt)[250], :]
scatter(X_tsne[:, 1], X_tsne[:, 2]; marker_z = vec(x' * P_sp) - x, color = :bwr, clim = (-0.01, 0.01))
