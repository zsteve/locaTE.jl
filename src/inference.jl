function get_MI_undir(X::AbstractMatrix, p::AbstractSparseVector, genes_i::Vector{Int}, genes_j::Vector{Int})
    # undirected mutual information
    @assert length(genes_i) == length(genes_j)
    mi = zeros(length(genes_i))
    for k = 1:length(genes_i)
        mi[k] = genes_i[k] == genes_j[k] ? 0 : get_mutual_information(discretized_joint_distribution_undir(p, X, genes_i[k], genes_j[k]))
    end
    mi
end

function get_MI_undir(X::AbstractMatrix, prod::AbstractSparseMatrix, genes_i::Vector{Int}, genes_j::Vector{Int})
    # undirected mutual information
    @assert length(genes_i) == length(genes_j)
    mi = zeros(length(genes_i))
    for k = 1:length(genes_i)
        mi[k] = genes_i[k] == genes_j[k] ? 0 : get_mutual_information(discretized_joint_distribution_undir(prod, X, genes_i[k], genes_j[k]))
    end
    mi
end

#=
function get_MI(X::AbstractMatrix, coupling_fw::AbstractSparseMatrix, coupling_rev::AbstractSparseMatrix, genes_prev::Vector{Int}, genes_next::Vector{Int}; rev = false)
    @assert length(genes_prev) == length(genes_next)
    mi_fwd = zeros(length(genes_prev))
    mi_rev = zeros(length(genes_prev))
    for j = 1:length(genes_prev)
        mi_fwd[j] = get_conditional_mutual_information(discretized_joint_distribution(coupling_fw, X, X, genes_prev[j], genes_next[j]))
        if rev
            mi_rev[j] = get_conditional_mutual_information(discretized_joint_distribution(coupling_rev, X, X, genes_next[j], genes_prev[j]))
        end
    end
    mi_fwd, mi_rev
end
=# 

function get_MI(X0::AbstractMatrix, X1::AbstractMatrix, coupling::AbstractSparseMatrix, genes_prev::Vector{Int}, genes_next::Vector{Int}; kwargs...)
    @assert length(genes_prev) == length(genes_next)
    mi = zeros(length(genes_prev))
    # construct row and col indices/maps 
    row_idxs = findnz(sum(coupling; dims = 2))[1]
    row_map = similar(row_idxs, Int64, size(X0, 1))
    row_map[row_idxs] .= collect(1:length(row_idxs))
    col_idxs = findnz(sum(coupling; dims = 1))[2]
    col_map = similar(col_idxs, Int64, size(X0, 1))
    col_map[col_idxs] .= collect(1:length(col_idxs))
    # 
    for j = 1:length(genes_prev)
        try
            mi[j] = get_conditional_mutual_information(discretized_joint_distribution(coupling, X0, X1, genes_prev[j], genes_next[j], row_idxs, col_idxs, row_map, col_map; kwargs...))
        catch e
            mi[j] = 0
        end
    end
    mi
end

function CLR(x)
    [0.5*sqrt.(relu(zscore(x[i, :])[j]).^2 + relu(zscore(x[:, j])[i]).^2) for i = 1:size(x, 1), j = 1:size(x, 2)]
end

function wCLR(x)
    [0.5*sqrt.(relu(zscore(x[i, :])[j]).^2 + relu(zscore(x[:, j])[i]).^2)*x[i, j] for i = 1:size(x, 1), j = 1:size(x, 2)]
end

function compute_coupling(X::AbstractMatrix, i::Int, R::AbstractSparseMatrix)
    pi = ((collect(1:size(X, 1)) .== i)'*1.0) * R 
    sparse(reshape(pi, :, 1)) .* R
end

function compute_coupling(X::AbstractMatrix, i::Int, P::AbstractSparseMatrix, R::AbstractSparseMatrix)
    pi = ((collect(1:size(X, 1)) .== i)'*1.0) * R 
    sparse(reshape(pi, :, 1)) .* P
end

function compute_coupling(X::AbstractMatrix, i::Int, P::AbstractSparseMatrix, QT::AbstractSparseMatrix, R::AbstractSparseMatrix)
    # given: Q a transition matrix t -> t-1; P a transition matrix t -> t+1
    # and π a distribution at time t
    # computes coupuling on (t-1, t+1) as Q'(πP)
    pi = ((collect(1:size(X, 1)) .== i)'*1.0) * R 
    QT * (sparse(reshape(pi, :, 1)) .* P)
end

apply_wclr(A, n_genes) = hcat(map(x -> vec(wCLR(reshape(x, n_genes, n_genes))), eachrow(A))...)'
apply_clr(A, n_genes) = hcat(map(x -> vec(CLR(reshape(x, n_genes, n_genes))), eachrow(A))...)'
