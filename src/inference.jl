function construct_index_maps(coupling)
    # construct row and col indices/maps
    w_row = sum(coupling; dims = 2)
    row_idxs = findnz(w_row)[1]
    row_map = similar(row_idxs, Int64, size(coupling, 1))
    row_map[row_idxs] .= collect(1:length(row_idxs))
    w_col = sum(coupling; dims = 1)
    col_idxs = findnz(w_col)[2]
    col_map = similar(col_idxs, Int64, size(coupling, 1))
    col_map[col_idxs] .= collect(1:length(col_idxs))
    return row_idxs, row_map, col_idxs, col_map
end

"""
    get_MI(X::AbstractMatrix, coupling::AbstractSparseMatrix, genes_prev::Vector{Int}, genes_next::Vector{Int}; disc = nothing, kwargs...)

Computes the TE score for all pairs of genes in `genes_prev, genes_next` from cell-by-gene expression matrix `X` under `coupling`.

"""
function get_MI(X::AbstractMatrix, coupling::AbstractSparseMatrix, genes_prev::Vector{Int}, genes_next::Vector{Int}; disc = nothing, kwargs...)
    @assert length(genes_prev) == length(genes_next)
    mi = zeros(length(genes_prev))
    row_idxs, row_map, col_idxs, col_map = construct_index_maps(coupling)
    # discretize each gene separately
    disc_prev = disc === nothing ? [discretization(X[row_idxs, i]; kwargs...) for i = 1:size(X, 2)] : [(x[1], x[2][row_idxs]) for x in disc]
    disc_next = disc === nothing ? [discretization(X[col_idxs, i]; kwargs...) for i = 1:size(X, 2)] : [(x[1], x[2][col_idxs]) for x in disc]
    for j = 1:length(genes_prev)
        try
            mi[j] = get_conditional_mutual_information(discretized_joint_distribution(coupling, X, genes_prev[j], genes_next[j], row_idxs, col_idxs, row_map, col_map, disc_prev, disc_next; kwargs...))
        catch e
            mi[j] = 0
        end
    end
    mi
end

"""
    CLR(x)

Compute CLR filtering of gene expression matrix `x`.

"""
function CLR(x)
    [0.5*sqrt.(relu(zscore(x[i, :])[j]).^2 + relu(zscore(x[:, j])[i]).^2) for i = 1:size(x, 1), j = 1:size(x, 2)]
end

"""
    wCLR(x)

Compute weighted CLR filtering of gene expression matrix `x`.

"""
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
