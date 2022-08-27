function discretization(x::AbstractVector; alg = DiscretizeUniformWidth(:scott))
    be = binedges(alg, x)
    disc = LinearDiscretizer(be)
    bi = encode(disc, x)
    return be, bi
end

function discretizations_bulk(X::AbstractMatrix; alg = DiscretizeBayesianBlocks())
    binedges_all = [binedges(alg, x) for x in eachcol(X)]
    discretizers_all = map(LinearDiscretizer, binedges_all)
    # counts_all = [get_discretization_counts(d, x) for (d, x) in zip(discretizers_all, eachcol(X))]
    binids_all = [encode(discretizers_all[i], X[:, i]) for i = 1:size(X, 2)]
    return discretizers_all, binedges_all, binids_all
end

#=
function discretized_joint_distribution(prod::AbstractMatrix, X::AbstractMatrix, i::Int, j::Int, binids_i::Vector{Int}, binids_j::Vector{Int}, binedges_i::AbstractVector, binedges_j::AbstractVector)
    # computes the discrete joint distribution of 
    # (X[i], X_next[j], X[j])
    π_genes = zeros(length(binedges_i), length(binedges_j), length(binedges_j))
    @inbounds for m = 1:size(X, 1)
        @inbounds for n = 1:size(X, 1)
            π_genes[binids_i[m], binids_j[n], binids_j[m]] += prod[m, n] 
        end
    end
    return π_genes
end
=#

function discretized_joint_distribution(prod::AbstractSparseMatrix, X0::AbstractMatrix, X1::AbstractMatrix, i::Int, j::Int; alg = DiscretizeUniformWidth(:scott))
    row_idxs = findnz(sum(prod; dims = 2))[1]
    col_idxs = findnz(sum(prod; dims = 1))[2]
    binedges_i_prev, binids_i_prev = discretization(X0[row_idxs, i]; alg = alg)
    binedges_j_next, binids_j_next = discretization(X1[col_idxs, j]; alg = alg)
    binedges_j_prev, binids_j_prev = discretization(X0[row_idxs, j]; alg = alg)
    discretized_joint_distribution(prod[row_idxs, :][:, col_idxs], 
                                    binids_i_prev, binids_j_next, binids_j_prev,
                                    binedges_i_prev, binedges_j_next, binedges_j_prev)
end

function discretized_joint_distribution_undir(prod::AbstractSparseMatrix, X::AbstractMatrix, i::Int, j::Int; alg = DiscretizeUniformWidth(:scott))
    row_idxs = findnz(sum(prod; dims = 2))[1]
    col_idxs = findnz(sum(prod; dims = 1))[2]
    binedges_i, binids_i = discretization(X[row_idxs, i]; alg = alg)
    binedges_j, binids_j = discretization(X[col_idxs, j]; alg = alg)
    discretized_joint_distribution_undir(prod[row_idxs, :][:, col_idxs],
                                        binids_i, binids_j,
                                        binedges_i, binedges_j)
end

function discretized_joint_distribution_undir(p::AbstractSparseVector, X::AbstractMatrix, i::Int, j::Int; alg = DiscretizeUniformWidth(:scott))
    idxs = findnz(p)[1]
    binedges_i, binids_i = discretization(X[idxs, i]; alg = alg)
    binedges_j, binids_j = discretization(X[idxs, j]; alg = alg)
    discretized_joint_distribution_undir(p[idxs],
                                        binids_i, binids_j,
                                        binedges_i, binedges_j)
end

function discretized_joint_distribution(prod::AbstractSparseMatrix, 
                                        binids_i_prev::Vector{Int}, binids_j_next::Vector{Int}, binids_j_prev::Vector{Int}, 
                                        binedges_i_prev::AbstractVector, binedges_j_next::AbstractVector, binedges_j_prev::AbstractVector)
    # computes the discrete joint distribution of 
    # (X[i], X_next[j], X[j])
    π_genes = zeros(length(binedges_i_prev)-1, length(binedges_j_next)-1, length(binedges_j_prev)-1) # this should be relatively small...
    for (m, n, p) in zip(findnz(prod)...)
            π_genes[binids_i_prev[m], binids_j_next[n], binids_j_prev[m]] += p
    end
    return π_genes
end

function discretized_joint_distribution_undir(p::AbstractSparseVector,
                                             binids_i::Vector{Int}, binids_j::Vector{Int},
                                             binedges_i::AbstractVector, binedges_j::AbstractVector)
    # compute the discrete joint distribution of (X[i], X[j])
    π_genes = zeros(length(binedges_i)-1, length(binedges_j)-1) # this should be relatively small...
    for (m, q) in zip(findnz(p)...)
            π_genes[binids_i[m], binids_j[m]] += q
    end
    return π_genes
end

function discretized_joint_distribution_undir(prod::AbstractSparseMatrix,
                                             binids_i::Vector{Int}, binids_j::Vector{Int},
                                             binedges_i::AbstractVector, binedges_j::AbstractVector)
    # compute the discrete joint distribution of (X[i], X[j])
    π_genes = zeros(length(binedges_i)-1, length(binedges_j)-1) # this should be relatively small...
    for (m, n, p) in zip(findnz(prod)...)
            π_genes[binids_i[m], binids_j[n]] += p
    end
    return π_genes
end
