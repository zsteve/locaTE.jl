using Pkg
Pkg.activate(".")
using Revise
Pkg.develop(path = "../../")
using NPZ
using OptimalTransport
using SparseArrays
import locaTE as lTE;
import NNlib: relu
import LinearAlgebra

DATA_DIR = "../simulated/data"
# DATA_DIR="data"
# X = convert(Matrix{Float32}, relu.(npzread(joinpath(DATA_DIR, "X.npy")) .- 1e-2))
X = convert(Matrix{Float32}, relu.(npzread(joinpath(DATA_DIR, "X.npy")) .- 10^(-0.5)))
X_pca = npzread(joinpath(DATA_DIR, "X_pca.npy"))
# P = npzread(joinpath(DATA_DIR, "P_statot.npy"))
# P = npzread(joinpath(DATA_DIR, "P_kappavelo.npy"))
P = npzread(joinpath(DATA_DIR, "P_velo_dot.npy"))
R = npzread(joinpath(DATA_DIR, "R.npy"))

k = 3
Q = lTE.to_backward_kernel(P)
P_sp = sparse((P^k))
QT_sp = sparse((Q^k)')
R_sp = sparse(R); 

L = lTE.construct_normalized_laplacian(X_pca, 25);

disc = lTE.discretizations_bulk(X);
targets, regulators = 1:size(X, 2), 1:size(X, 2)
gene_idxs = vcat([[j, i]' for i in targets for j in regulators]...);
clusters = LinearAlgebra.I(size(P, 1))
clusters_norm = convert(Matrix{eltype(P)}, clusters)
clusters_norm ./= sum(clusters_norm; dims = 1);

i, j, k = (903, 1, 4)
coupling=lTE.compute_coupling(X, clusters_norm[:, i], P_sp, QT_sp, R_sp)
_I, _J, _V = findnz(coupling);
disc_max_size = maximum(map(x -> length(x[1]) - 1, disc))
p = zeros(disc_max_size, disc_max_size, disc_max_size)

disc_prev =
	disc === nothing ?
		[discretization(X[:, i]; kwargs...) for i = 1:size(X, 2)] :
		[(x[1], x[2]) for x in disc];
disc_next =
	disc === nothing ?
		[discretization(X[:, i]; kwargs...) for i = 1:size(X, 2)] :
		[(x[1], x[2]) for x in disc];

lTE.discretized_joint_distribution!(
    p, 
    _I, _J, _V,
    X,
    j,
    k,
    disc_prev,
    disc_next;
)

## try CUDA
using CUDA
N_blocks=1
joint_cache = lTE.get_joint_cache(length(regulators) ÷ N_blocks, length(targets) ÷ N_blocks, disc_max_size);
ids_cu = hcat(map(x -> x[2], disc)...) |> cu
# Copy transition matrices and neighbourhood kernel to CUDA device
P_cu = cu(P)
QT_cu = cu(Array(QT_sp))
R_cu = cu(R)
clusters_norm = convert(Matrix{eltype(P_cu)}, clusters)
clusters_norm ./= sum(clusters_norm; dims = 1)
clusters_norm_cu = cu(clusters_norm)
TE = CuArray{eltype(joint_cache)}(undef, (size(clusters, 2), length(regulators), length(targets)))
I, J, V = lTE.getcoupling_sparse(clusters_norm[:, i], P_sp, QT_sp, R_sp)
for ((N_x, N_y), (offset_x, offset_y)) in lTE.getblocks(length(regulators), length(targets), N_blocks, N_blocks)
    lTE.get_MI!(
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

vec(sum(joint_cache[j, k, :, :, :]; dims=(1, 2)))
vec(sum(p; dims=(1,2)))

vec(sum(joint_cache[j, k, :, :, :]; dims=(1, 3)))
vec(sum(p; dims=(1,3)))

sum(abs.(p - Array(joint_cache[j, k, :, :, :])))

using InformationMeasures
get_conditional_mutual_information(Array(joint_cache[j, k, :, :, :]))
TE[i,j,k]

q = Array(joint_cache[j, k, :, :, :])

using LogExpFunctions 

import InformationMeasures.get_conditional_mutual_information
function get_conditional_mutual_information(xyz; estimator = "maximum_likelihood", base = exp(1), probabilities = false, lambda = nothing, prior = 1)
	probabilities_xyz = probabilities ? xyz : get_probabilities(estimator, xyz, lambda = lambda, prior = prior)
	probabilities_xz = sum(probabilities_xyz, dims = 2)
	probabilities_yz = sum(probabilities_xyz, dims = 1)
	probabilities_z = sum(probabilities_xz, dims = 1) # xz not a typo
	entropy_xyz = apply_entropy_formula(probabilities_xyz, base)
	entropy_xz = apply_entropy_formula(probabilities_xz, base)
	entropy_yz = apply_entropy_formula(probabilities_yz, base)
	entropy_z = apply_entropy_formula(probabilities_z, base)
	@info entropy_xz, entropy_yz, entropy_xyz, entropy_z
	# TODO: either deprecate or add warning about inappropriate use
	if estimator == "miller_madow"
		entropy_xyz += (count(!iszero, probabilities_xyz) - 1) / (2 * length(probabilities_xyz))
		entropy_xz += (count(!iszero, probabilities_xz) - 1) / (2 * length(probabilities_xz))
		entropy_yz += (count(!iszero, probabilities_yz) - 1) / (2 * length(probabilities_yz))
		entropy_z += (count(!iszero, probabilities_z) - 1) / (2 * length(probabilities_z))
	end
	return apply_conditional_mutual_information_formula(entropy_xz, entropy_yz, entropy_xyz, entropy_z)
end

H_xz = mapreduce(xlogx, +, sum(q; dims = 2); dims = (1, 3))
H_yz = mapreduce(xlogx, +, sum(q; dims = 1); dims = (2, 3))
H_xyz = dropdims(mapreduce(xlogx, +, q; dims = (1,2,3)))
H_z = dropdims(mapreduce(xlogx, +, sum(q; dims = (1,2)); dims = 3))
-H_xz - H_yz + H_xyz + H_z

# H_xz = dropdims(mapreduce(xlogx, +, sum(joint_probs; dims = 4); dims = (3, 5)))
# H_yz = dropdims(mapreduce(xlogx, +, sum(joint_probs; dims = 3); dims = (4, 5)))
# H_xyz = dropdims(mapreduce(xlogx, +, joint_probs; dims = (3, 4, 5)))
# H_z = dropdims(mapreduce(xlogx, +, sum(joint_probs; dims = (3, 4)); dims = 5))

get_conditional_mutual_information(p) 
get_conditional_mutual_information(q) 


using BenchmarkTools 
import LinearAlgebra

i = 1
disc = lTE.discretizations_bulk(X);
targets, regulators = 1:size(X, 2), 1:size(X, 2)
gene_idxs = vcat([[j, i]' for i in targets for j in regulators]...);
clusters = LinearAlgebra.I(size(P, 1))
clusters_norm = convert(Matrix{eltype(P)}, clusters)
clusters_norm ./= sum(clusters_norm; dims = 1);

TE = lTE.estimate_TE(
    X,
    1:size(X, 2),
    1:size(X, 2),
    P_sp,
    QT_sp,
    R_sp,
);

coupling=lTE.compute_coupling(X, clusters_norm[:, i], P_sp, QT_sp, R_sp)
# 10s, 5.26gb alloc
# 5s 38mb 
@time lTE.get_MI(X, coupling, gene_idxs[:, 1], gene_idxs[:, 2]; disc = disc); 

@time lTE.get_MI(X, lTE.compute_coupling(X, clusters_norm[:, i], P_sp, QT_sp, R_sp), gene_idxs[:, 1], gene_idxs[:, 2]; disc = disc); 
# @time my_get_MI(X, lTE.compute_coupling(X, clusters_norm[:, i], P_sp, QT_sp, R_sp), gene_idxs[:, 1], gene_idxs[:, 2]; disc = disc); 

TE_a = lTE.get_MI(X, lTE.compute_coupling(X, clusters_norm[:, i], P_sp, QT_sp, R_sp), gene_idxs[:, 1], gene_idxs[:, 2]; disc = disc)
TE_b = my_get_MI(X, lTE.compute_coupling(X, clusters_norm[:, i], P_sp, QT_sp, R_sp), gene_idxs[:, 1], gene_idxs[:, 2]; disc = disc)

sum(abs.(TE_a[1:10] - TE_b[1:10]))

TE = lTE.estimate_TE(
    X,
    1:size(X, 2),
    1:size(X, 2),
    P_sp,
    QT_sp,
    R_sp,
);

# 3ms, 10.42mb
@benchmark lTE.compute_coupling($X, $(clusters_norm[:, i]), $P_sp, $QT_sp, $R_sp)

# ~20us, 38kb
@benchmark lTE.construct_index_maps(coupling)

# row_idxs, row_map, col_idxs, col_map = lTE.construct_index_maps(coupling)

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

