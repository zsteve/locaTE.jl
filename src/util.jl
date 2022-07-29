symm(A::AbstractMatrix) = max.(A, A') # 0.5*(A + A')
symm(x::AbstractVector, n) = symm(reshape(x, n, n))
symm_row(A::AbstractMatrix, n) = hcat([vec(symm(x, n)) for x in eachrow(A)]...)'
cartesian_to_index(i, j; N) = N*(j-1)+i
