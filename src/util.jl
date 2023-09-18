"""
    symm(A::AbstractMatrix)

Symmetrize matrix by applying `max` elementwise.
"""
symm(A::AbstractMatrix) = max.(A, A') # 0.5*(A + A')

"""
    symm(x::AbstractVector, n)

Symmetrize flattened matrix by applying `max` elementwise.
"""
symm(x::AbstractVector, n) = symm(reshape(x, n, n))

"""
    symm_row(A::AbstractMatrix, n)

Symmetrize array of flattened matrices, i.e. of dimensions `(m, n^2)`.
"""
symm_row(A::AbstractMatrix, n) = hcat([vec(symm(x, n)) for x in eachrow(A)]...)'

cartesian_to_index(i, j; N) = N * (j - 1) + i

"""
    aupr(p::AbstractVector, r::AbstractVector)

Compute AUPR from a vector of precision `p` and recall `r` rates at different thresholds. 
"""
aupr(p::AbstractVector, r::AbstractVector) = dot(p[1:end-1], abs.(r[2:end] - r[1:end-1]))

"""
    auroc(tp::AbstractVector, fp::AbstractVector)

Compute AUROC from a vector of true positive rates `tp` and false positive rates `fp` at different thresholds.
"""
auroc(tp::AbstractVector, fp::AbstractVector) = aupr(tp, fp)


"""
    ep(p::AbstractVector, r::AbstractVector; f = 0.1)

Compute early precision (EP) from a vector of precision `p` and recall `r` rates at different thresholds,
for `r ≤ f`.
"""
function ep(p::AbstractVector, r::AbstractVector; f = 0.1)
    ind = r .<= f
    aupr(p[ind], r[ind]) / f
end


"""
    prec_rec_rate(J::AbstractMatrix, Z::AbstractMatrix, q::Real; J_thresh = 0.5)

Compute precision and recall rates for a ground truth matrix `J`, score matrix `Z`, threshold `q ∈ [0, 1]`.
Entries of `J` such that `abs.(J) .> J_thresh` are treated as true edges.
"""
function prec_rec_rate(J::AbstractMatrix, Z::AbstractMatrix, q::Real; J_thresh = 0.5)
    edges_true = (abs.(J) .> J_thresh)
    edges_infer = (Z .> (minimum(Z) + q * (maximum(Z) - minimum(Z))))
    tp = sum(edges_true .& edges_infer)
    fp = sum(.!edges_true .& edges_infer)
    fn = sum(edges_true .& .!edges_infer)
    return [tp / (tp + fp), tp / (tp + fn)]
end

"""
    prec_rec_rate(J::AbstractMatrix, Z::AbstractMatrix, Nq::Integer; kwargs...)

Compute vectors of precision and recall rates for `Nq` uniformly spaced discrimination thresholds. 
"""
function prec_rec_rate(J::AbstractMatrix, Z::AbstractMatrix, Nq::Integer; kwargs...)
    hcat(
        [
            prec_rec_rate(J, Z, q; kwargs...) for
            q in range(0 - 1e-6, 1 + 1e-6; length = Nq)
        ]...,
    )'
end

"""
    tp_fp_rate(J::AbstractMatrix, Z::AbstractMatrix, q::Real; J_thresh = 0.5)

Compute true positive and false positive rates for a ground truth matrix `J`, score matrix `Z`, threshold `q ∈ [0, 1]`.
Entries of `J` such that `abs.(J) .> J_thresh` are treated as true edges.
"""
function tp_fp_rate(J::AbstractMatrix, Z::AbstractMatrix, q::Real; J_thresh = 0.5)
    edges_true = (abs.(J) .>= J_thresh)
    edges_infer = (Z .> (minimum(Z) + q * (maximum(Z) - minimum(Z))))
    tp = sum(edges_true .& edges_infer)
    fp = sum(.!edges_true .& edges_infer)
    fn = sum(edges_true .& .!edges_infer)
    tn = sum(.!edges_true .& .!edges_infer)
    return [tp / (tp + fn), fp / (fp + tn)]
end

"""
    tp_fp_rate(J::AbstractMatrix, Z::AbstractMatrix, Nq::Integer; kwargs...)

Compute vectors of true positive and false positive rates for `Nq` uniformly spaced discrimination thresholds. 
"""
function tp_fp_rate(J::AbstractMatrix, Z::AbstractMatrix, Nq::Integer; kwargs...)
    hcat(
        [tp_fp_rate(J, Z, q; kwargs...) for q in range(0 - 1e-6, 1 + 1e-6; length = Nq)]...,
    )'
end

function cdf_norm(x::AbstractArray, A::AbstractMatrix; fits_reg_all = nothing, fits_targ_all = nothing)
    A_norm = zero(x)
    if fits_reg_all === nothing fits_reg_all = [fit(Gamma, vcat(A[i, 1:i-1], A[i, i+1:end])) for i = 1:size(A, 1)] end
    if fits_targ_all === nothing fits_targ_all = [fit(Gamma, vcat(A[1:j-1, j], A[j+1:end, j])) for j = 1:size(A, 2)] end
    for (i, j) in Iterators.product(1:size(A, 1), 1:size(A, 2))
        A_norm[i, j] = (cdf(fits_reg_all[i], x[i, j]) + cdf(fits_targ_all[j], x[i, j])) / 2
    end
    return A_norm
end

function apply_cdf_norm(x::AbstractArray, A::AbstractMatrix)
    A_norm = zero(x)
    fits_reg_all = [fit(Gamma, vcat(A[i, 1:i-1], A[i, i+1:end])) for i = 1:size(A, 1)]
    fits_targ_all = [fit(Gamma, vcat(A[1:j-1, j], A[j+1:end, j])) for j = 1:size(A, 2)]
    @showprogress for (i, _x) in enumerate(eachrow(x))
        A_norm[i, :] .= vec(cdf_norm(reshape(_x, size(A)...), A;
                                    fits_reg_all = fits_reg_all, fits_targ_all = fits_targ_all))
    end
    A_norm
end

pmean(x, p; kwargs...) = mean(x .^ p; kwargs...).^(1/p)

# From https://github.com/JuliaPlots/Plots.jl/blob/master/src/recipes.jl 
# fallback function for finding non-zero elements of non-sparse matrices
# function findnz(A::AbstractMatrix)
#     keysnz = findall(!iszero, A)
#     rs = map(k -> k[1], keysnz)
#     cs = map(k -> k[2], keysnz)
#     zs = Array(A[keysnz])
#     rs, cs, zs
# end
