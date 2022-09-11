function discretization(x::AbstractVector; alg = DiscretizeUniformWidth(:scott))
    try
        be = binedges(alg, x)
        disc = LinearDiscretizer(be)
        bi = encode(disc, x)
        return be, bi
    catch e
        @info e
        return [], zero.(x)
    end
end

function discretizations_bulk(X::AbstractMatrix; alg = DiscretizeBayesianBlocks())
    binedges_all = [binedges(alg, x) for x in eachcol(X)]
    discretizers_all = map(LinearDiscretizer, binedges_all)
    # counts_all = [get_discretization_counts(d, x) for (d, x) in zip(discretizers_all, eachcol(X))]
    binids_all = [encode(discretizers_all[i], X[:, i]) for i = 1:size(X, 2)]
    # return discretizers_all, binedges_all, binids_all
    return zip(binedges_all, binids_all)
end

function discretized_joint_distribution(prod::AbstractSparseMatrix, X::AbstractMatrix, i::Int, j::Int, row_idxs, col_idxs, row_map, col_map, disc_prev, disc_next; alg = DiscretizeUniformWidth(:scott))
    binedges_i_prev, binids_i_prev = disc_prev[i]
    binedges_j_next, binids_j_next = disc_next[j] 
    binedges_j_prev, binids_j_prev = disc_prev[j] 
    discretized_joint_distribution(prod, 
                                    binids_i_prev, binids_j_next, binids_j_prev,
                                    binedges_i_prev, binedges_j_next, binedges_j_prev,
                                    row_map, col_map)
end

function discretized_joint_distribution(prod::AbstractSparseMatrix, 
                                        binids_i_prev::Vector{Int}, binids_j_next::Vector{Int}, binids_j_prev::Vector{Int}, 
                                        binedges_i_prev::AbstractVector, binedges_j_next::AbstractVector, binedges_j_prev::AbstractVector,
                                        row_map, col_map)
    # computes the discrete joint distribution of 
    # (X[i], X_next[j], X[j])
    π_genes = zeros(length(binedges_i_prev)-1, length(binedges_j_next)-1, length(binedges_j_prev)-1) # this should be relatively small...
    for (m, n, p) in zip(findnz(prod)...)
        π_genes[binids_i_prev[row_map[m]], binids_j_next[col_map[n]], binids_j_prev[row_map[m]]] += p
    end
    return π_genes
end
