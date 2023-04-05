function discretization(x::AbstractVector; alg::DiscretizationAlgorithm = DiscretizeUniformWidth(:scott))
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

"""
    discretizations_bulk(X::AbstractMatrix; alg = DiscretizeBayesianBlocks())

Discretize each column of `X` using algorithm `alg`. 
"""
function discretizations_bulk(X::AbstractMatrix; alg::DiscretizationAlgorithm = DiscretizeBayesianBlocks())
    binedges_all = [binedges(alg, x) for x in eachcol(X)]
    discretizers_all = map(LinearDiscretizer, binedges_all)
    # counts_all = [get_discretization_counts(d, x) for (d, x) in zip(discretizers_all, eachcol(X))]
    binids_all = [encode(discretizers_all[i], X[:, i]) for i = 1:size(X, 2)]
    # return discretizers_all, binedges_all, binids_all
    return zip(binedges_all, binids_all)
end

function _discretized_joint_distribution!(
    π_genes::AbstractArray, 
    I::AbstractVector{Int}, J::AbstractVector{Int}, V::AbstractVector,
    X::AbstractMatrix,
    i::Int,
    j::Int,
    disc_prev,
    disc_next;
)
    binedges_i_prev, binids_i_prev = disc_prev[i]
    binedges_j_next, binids_j_next = disc_next[j]
    binedges_j_prev, binids_j_prev = disc_prev[j]
    _discretized_joint_distribution!(
        π_genes, 
        I, J, V,
        binids_i_prev,
        binids_j_next,
        binids_j_prev,
        binedges_i_prev,
        binedges_j_next,
        binedges_j_prev,
    )
end

function _discretized_joint_distribution!(
    π_genes::AbstractArray, 
    I::AbstractVector{Int}, J::AbstractVector{Int}, V::AbstractVector,
    binids_i_prev::AbstractVector{Int},
    binids_j_next::AbstractVector{Int},
    binids_j_prev::AbstractVector{Int},
    binedges_i_prev::AbstractVector,
    binedges_j_next::AbstractVector,
    binedges_j_prev::AbstractVector,
)
    # computes the discrete joint distribution of 
    # (X[i], X_next[j], X[j])
    # π_genes = zeros(
    #     length(binedges_i_prev) - 1,
    #     length(binedges_j_next) - 1,
    #     length(binedges_j_prev) - 1,
    # ) # this should be relatively small...
    # for (m, n, p) in zip(I, J, V)
    @inbounds begin
        for i = 1:length(I)
            m = I[i]
            n = J[i]
            p = V[i]
            # π_genes[
            #     binids_i_prev[row_map[m]],
            #     binids_j_next[col_map[n]],
            #     binids_j_prev[row_map[m]],
            # ] += p
            π_genes[
                binids_i_prev[m],
                binids_j_next[n],
                binids_j_prev[m],
            ] += p
        end
    end
    # return π_genes
end
