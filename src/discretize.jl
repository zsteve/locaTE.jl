"""
    discretization(x::AbstractVector; alg::DiscretizationAlgorithm = DiscretizeBayesianBlocks())

Discretize vector `x` using algorithm `alg`. 

"""

function discretization(x::AbstractVector; alg::DiscretizationAlgorithm = DiscretizeBayesianBlocks())
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
    binids_all = [encode(discretizers_all[i], X[:, i]) for i = 1:size(X, 2)]
    return zip(binedges_all, binids_all)
end

"""
    discretized_joint_distribution!(
        π_genes::AbstractArray, 
        I::AbstractVector{Int}, J::AbstractVector{Int}, V::AbstractVector,
        i::Int,
        j::Int,
        disc_prev,
        disc_next;
    )

Form in `π_genes` the discretized joint distribution for a pair of genes `(i, j)` under sparse
coupling represented as `(I, J, V)`. `disc_prev` and `disc_next` are discretizations for genes 
at the previous and next cell state respectively. 
`π_genes` corresponds to `(X[i], Y[j], X[j])` where `X, Y` are the previous and next states 
of the cell state under the coupling. 

"""
function discretized_joint_distribution!(
    π_genes::AbstractArray, 
    I::AbstractVector{Int}, J::AbstractVector{Int}, V::AbstractVector,
    i::Int,
    j::Int,
    disc_prev,
    disc_next;
)
    binedges_i_prev, binids_i_prev = disc_prev[i]
    binedges_j_next, binids_j_next = disc_next[j]
    binedges_j_prev, binids_j_prev = disc_prev[j]
    discretized_joint_distribution!(
        π_genes, 
        I, J, V,
        binids_i_prev,
        binids_j_next,
        binids_j_prev,
    )
end

"""

    discretized_joint_distribution!(
        π_genes::AbstractArray, 
        I::AbstractVector{Int}, J::AbstractVector{Int}, V::AbstractVector,
        binids_i_prev::AbstractVector{Int},
        binids_j_next::AbstractVector{Int},
        binids_j_prev::AbstractVector{Int},
    )

Implementation of accumulation step for `discretized_joint_distribution!`. 

"""
function discretized_joint_distribution!(
    π_genes::AbstractArray, 
    I::AbstractVector{Int}, J::AbstractVector{Int}, V::AbstractVector,
    binids_i_prev::AbstractVector{Int},
    binids_j_next::AbstractVector{Int},
    binids_j_prev::AbstractVector{Int},
)
    # computes the discrete joint distribution of 
    # (X[i], X_next[j], X[j])
    @inbounds begin
        for (m, n, p) in zip(I, J, V)
            π_genes[
                binids_i_prev[m],
                binids_j_next[n],
                binids_j_prev[m],
            ] += p
        end
    end
end
