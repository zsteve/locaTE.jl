"""
    prox_l1(x, λ)

Proximal operator for the L1 norm with weight `λ`, ``x \\mapsto λ\\|x\\|_1``. 
"""
prox_l1(x, λ) = sign(x) * relu(abs(x) - λ)

"""
    fitsp(G::AbstractMatrix, L::AbstractMatrix, α; ρ = 0.05, λ1 = 25.0, λ2 = 0.075, maxiter = 2500)

Denoise TE scores by solving the *weighted* L1-L2 regularized regression problem
```math
        \\min_{X} \\frac{1}{2} \\sum_{i = 1}^{N} \\alpha_{i} \\| X_i - G_i \\|_2^2
        + \\frac{λ_1}{2} \\operatorname{tr}(X^\\top L X) + λ_2 \\sum_{i = 1}^N \\alpha_i \\| X_i \\|_1.
```
"""
function fitsp(
    G::AbstractMatrix,
    L::AbstractMatrix,
    α;
    ρ = 0.05,
    λ1 = 25.0,
    λ2 = 0.075,
    maxiter = 2500,
)
    # scaling factors
    L_scaled = sqrt.(α) * L * sqrt.(α)
    X = similar(G)
    copy!(X, G)
    Z = similar(G)
    copy!(Z, G)
    W = zero(G)
    X_new = similar(X)
    Z_new = similar(Z)
    W_new = similar(W)
    ΔX, ΔZ, ΔW = 0, 0, 0
    @showprogress for iter = 1:maxiter
        # X_new = ((1+ρ)I + λ1*L) \ (G - ρ*(W-Z));
        X_new = (α + λ1 * L_scaled + ρ * I) \ (α * G + ρ * (Z - W))
        # Z_new = prox_l1.(X_new+W; λ = λ2/ρ);
        # for i = 1:size(L, 1)
        #     # Z_new = hcat([prox_l1.(X_new[i, :]+W[i, :], λ2*diag(α)[i]/ρ) for i = 1:size(L, 1)]...)';
        #     Z_new[i, :] .= prox_l1.(X_new[i, :]+W[i, :], λ2*diag(α)[i]/ρ)
        # end
        Z_new .= prox_l1.(X_new + W, λ2 * diag(α) / ρ)
        W_new = W + X_new - Z_new
        ΔX, ΔZ, ΔW = norm(X - X_new, Inf), norm(Z - Z_new, Inf), norm(W - W_new, Inf)
        copy!(X, X_new)
        copy!(Z, Z_new)
        copy!(W, W_new)
    end
    @info "ΔX = $(ΔX), ΔZ = $(ΔZ), ΔW = $(ΔW)"
    @info "tr(X'LX) = $(tr(X'*L_scaled*X)), 0.5|X-G|^2 = $(0.5*norm(X-G)), |X|1 = $(norm(X, 1))"
    Z
end

"""
    fitsp(G::AbstractMatrix, L::AbstractMatrix; ρ = 0.05, λ1 = 25.0, λ2 = 0.075, maxiter = 2500)

Denoise TE scores by solving the L1-L2 regularized regression problem
```math
        \\min_{X} \\frac{1}{2} \\sum_{i = 1}^{N} \\| X_i - G_i \\|_2^2
        + \\frac{λ_1}{2} \\operatorname{tr}(X^\\top L X) + λ_2 \\sum_{i = 1}^N \\| X_i \\|_1.
```
"""
function fitsp(
    G::AbstractMatrix,
    L::AbstractMatrix;
    ρ = 0.05,
    λ1 = 25.0,
    λ2 = 0.075,
    maxiter = 2500,
)
    # scaling factors
    X = similar(G)
    copy!(X, G)
    Z = similar(G)
    copy!(Z, G)
    W = zero(G)
    ΔX, ΔZ, ΔW = 0, 0, 0
    @showprogress for iter = 1:maxiter
        X_new = ((1 + ρ)I + λ1 * L) \ (G - ρ * (W - Z))
        Z_new = prox_l1.(X_new + W, λ2 / ρ)
        W_new = W + X_new - Z_new
        ΔX, ΔZ, ΔW = norm(X - X_new, Inf), norm(Z - Z_new, Inf), norm(W - W_new, Inf)
        X = X_new
        Z = Z_new
        W = W_new
    end
    @info "ΔX = $(ΔX), ΔZ = $(ΔZ), ΔW = $(ΔW)"
    @info "tr(X'LX) = $(tr(X'*L*X)), 0.5|X-G|^2 = $(0.5*norm(X-G)), |X|1 = $(norm(X, 1))"
    Z
end

"""
    fitnmf(G, L_all, L, H, k; α = 0, β = 0, λ = [0, 0], μ = [0, 0], iter = 500, print_iter = 50, initialize = :nndsvd, δ = 1e-5, dictionary = false, η = 1.0, U_init = nothing, V_init = nothing)

Regularized non-negative matrix factorization by solving the problem

```math
    \\min_{U, V} \\frac{1}{2} \\| UV^\\top - G \\|_2^2 + \\frac{α}{2} \\operatorname{tr}(VU^\\top L UV^\\top)
        - \\beta \\langle H, UV^\\top \\rangle + \\frac{λ_1}{2} \\operatorname{tr}(U^\\top K_1 U)
        +  μ_1 \\| U \\|_1 + \\frac{λ_2}{2} \\operatorname{tr}(V^\\top K_2 V) + μ_2 \\| V \\|_1.
```

`L_all` contains positive semidefinite (potentially sparse) matrices corresponding to ``[K_1, K_2]`` that act on the factor matrices,
while `L` is a positive semidefinite matrix acting on the low rank reconstruction.

A number of initializations are possible by setting the value of `initialize`: random (`:rand`),
nonnegative double singular value decomposition (`:nndsvd`, using the implementation [here](https://github.com/JuliaStats/NMF.jl)),
2 iterations of NMF (`:nmf`, using [this function](https://github.com/JuliaStats/NMF.jl)),
or manual initialization `U_init, V_init` (`:manual`).

Returns `U, V` and `trace` containing objective values. 
"""
function fitnmf(
    G,
    L_all,
    L,
    H,
    k;
    α = 0,
    β = 0,
    λ = [0, 0],
    μ = [0, 0],
    iter = 500,
    print_iter = 50,
    initialize = :nndsvd,
    δ = 1e-5,
    dictionary = false,
    η = 1.0,
    U_init = nothing,
    V_init = nothing,
)
    # use multiplicative updates
    U = similar(G, size(G, 1), k)
    U_new = similar(U)
    V = similar(G, size(G, 2), k)
    V_new = similar(V)
    tocu = (G isa CuArray ? CuArray : x -> x)
    tocpu = (G isa CuArray ? Array : x -> x)
    D = tocu(Diagonal(diag(L)))
    L = tocu(L)
    A = D - L
    D1 = tocu(Diagonal(diag(L_all[1])))
    D2 = tocu(Diagonal(diag(L_all[2])))
    L_all = map((G isa CuArray ? CuArray : x -> x), L_all)
    A1 = D1 - L_all[1]
    A2 = D2 - L_all[2]
    # initialize
    @info "Initializing NMF decomposition with $(initialize)"
    if initialize === :rand
        copy!(U, rand(size(U)...))
        copy!(V, rand(size(V)...))
    elseif initialize === :nndsvd
        U_init, V_init = NMF.nndsvd(tocpu(G), k)
        copy!(U, U_init)
        copy!(V, V_init')
    elseif initialize === :nmf
        tmp = nnmf(tocpu(G), k; alg = :multmse, maxiter = 2)
        copy!(U, tmp.W)
        copy!(V, tocu(tmp.H'))
    elseif initialize == :manual
        copy!(U, U_init)
        copy!(V, V_init)
    end
    if dictionary
        V .*= mean(sum(U; dims = 2))
        U ./= sum(U; dims = 2)
    end
    trace = []
    @showprogress for it = 1:iter
        function objective()
            Dict(
                :df => norm(U * V' - G, 2)^2 / 2,
                :smooth => α / 2 * tr(V * U' * L * (U * V')),
                :affine => -β * tr(H' * U * V'),
                :smooth_U => λ[1] / 2 * tr(U' * L_all[1] * U),
                :sp_U => μ[1] * norm(U, 1),
                :smooth_V => λ[2] / 2 * tr(V' * L_all[2] * V),
                :sp_V => μ[2] * norm(V, 1),
            )
        end
        U_new .=
            relu.(U .* (G * V + λ[1] * A1 * U + α * A * U * V' * V + β * H * V)) ./
            (α * D * U * V' * V + U * V' * V + λ[1] * (D1 * U) .+ μ[1] .+ δ)
        if dictionary
            U_new ./= sum(U_new; dims = 2)
        end
        V_new .=
            relu.(V .* (G' * U + λ[2] * A2 * V + α * V * U' * A * U + β * H' * U)) ./
            (α * V * U' * D * U + V * U' * U + λ[2] * (D2 * V) .+ μ[2] .+ δ)
        if (it % print_iter == 0)
            ΔU, ΔV = norm(U - U_new, Inf), norm(V - V_new, Inf)
            obj = objective()
            # @info obj
            l = sum([v for (k, v) in obj])
            # @info "iteration $(it), (ΔU, ΔV) = $((ΔU, ΔV)), L = $(l), objs = $(obj)"
            push!(trace, l)
        end
        # copy!(U, U_new)
        # copy!(V, V_new)
        U .= (1 - η) * U + η * U_new
        V .= (1 - η) * V + η * V_new
    end
    U, V, trace
end

"""
    fitntf(G, L, L_g, H, λ, μ, α, β, k; iter = 250, print_iter = 50, dictionary = false, δ = 1e-5, η = 1.0)

Regularized non-negative tensor factorization by solving the problem

```math
    \\min_{S, \\{ A^{(i)} \\}_{i = 1}^3} \\frac{1}{2} \\| X - G\\|_2^2 + \\frac{\\alpha}{2} \\operatorname{tr}(X_{(1)}^\\top L X_{(1)})
        - \\beta \\langle H, X\\rangle + \\sum_{i = 1}^3 \\frac{\\lambda_i}{2} \\operatorname{tr}((A^{(i)})^\\top L^{(i)} A^{(i)})
        + \\sum_{i = 1}^3 \\mu_i \\| A^{(i)} \\|_1. 
```
where for brevity ``X = S \\times_{i = 1}^3 A^{(i)}``. 

`L` contains positive semidefinite (potentially sparse) matrices corresponding to ``L^{(i)}`` in the above formula that act on the factor matrices,
and `L_g` corresponds to `L`, acting on the low rank reconstruction.

The decomposition is currently initialised using the [Tensorly](http://tensorly.org/) library, with 1 iteration of `non_negative_parafac` with `init = "svd"`.

Currently only optimises over the factor matrices while keeping `S` fixed (i.e. seeks a CP decomposition)
"""
function fitntf(
    G,
    L,
    L_g,
    H,
    λ,
    μ,
    α,
    β,
    k;
    iter = 250,
    print_iter = 50,
    dictionary = false,
    δ = 1e-5,
    η = 1.0,
)
    tocu = (G isa CuArray ? CuArray : x -> x)
    tocpu = (G isa CuArray ? Array : x -> x)
    # factor matrices
    A = [similar(G, size(G, i), k) for i = 1:length(size(G))]
    A_new = [similar(a) for a in A]
    # core tensor
    S = zeros([k for _ = 1:length(size(G))]...)
    for i = 1:k
        S[i, i, i] = 1.0
    end
    S = tocu(S)
    # initialization 
    factor_cp = tl_decomp.non_negative_parafac(
        tocpu(G),
        rank = k,
        init = "svd",
        n_iter_max = 1,
        random_state = 0,
    )
    for i = 1:length(A)
        copy!(A[i], factor_cp.factors[i])
    end
    # normalize factor A^{(1)} if we need dictionary interpretation
    if dictionary
        A[1] ./= sum(A[1]; dims = 2)
    end
    # 
    D = [tocu(Diagonal(diag(l))) for l in L]
    L = map(tocu, L)
    W = [d - l for (d, l) in zip(D, L)]
    D_g = tocu(Diagonal(diag(L_g)))
    L_g = tocu(L_g)
    W_g = D_g - L_g
    ΔA = zeros(length(A))
    trace = []
    @showprogress for it = 1:iter
        function objective()
            X = ttm(S, A, 1:length(A))
            X1 = tenmat(X, 1)
            Dict(
                :df => norm(X - G, 2)^2 / 2,
                :smooth => [λ[i] / 2 * tr(A[i]' * L[i] * A[i]) for i = 1:length(A)],
                :sparse => [μ[i] * norm(A[i], 1) for i = 1:length(A)],
                :smooth_mode1 => α / 2 * tr(X1' * L_g * X1),
                :affine => -β * sum(H .* X),
            )
        end
        for i = 1:length(A)
            inds = [j for j in range(length = length(A)) if j != i]
            Bi = tenmat(ttm(S, [Matrix(A[j]) for j in inds], inds), i)
            if i == 1
                A_new[i] .=
                    relu.(
                        A[i] .* (
                            tenmat(G, i) * Bi' +
                            λ[i] * W[i] * A[i] +
                            α * W_g * A[i] * Bi * Bi' +
                            β * tenmat(H, i) * Bi'
                        ) ./ (
                            A[i] * Bi * Bi' +
                            λ[i] * D[i] * A[i] +
                            α * D_g * A[i] * Bi * Bi' .+ μ[i] .+ δ
                        )
                    )
            else
                Btilde_plus = tenmat(
                    ttm(S, [Matrix((j == 1 ? D_g : tocu(I)) * A[j]) for j in inds], inds),
                    i,
                )
                Btilde_minus = tenmat(
                    ttm(S, [Matrix((j == 1 ? W_g : tocu(I)) * A[j]) for j in inds], inds),
                    i,
                )
                C_plus = (Btilde_plus * Bi' + Bi * Btilde_plus') / 2
                C_minus = (Btilde_minus * Bi' + Bi * Btilde_minus') / 2
                A_new[i] .=
                    relu.(
                        A[i] .* (
                            tenmat(G, i) * Bi' +
                            λ[i] * W[i] * A[i] +
                            α * A[i] * C_minus +
                            β * tenmat(H, i) * Bi'
                        ) ./ (
                            A[i] * Bi * Bi' + λ[i] * D[i] * A[i] + α * A[i] * C_plus .+
                            μ[i] .+ δ
                        )
                    )
            end
            # normalize first factor
            if dictionary && (i == 1)
                A_new[i] ./= sum(A_new[i]; dims = 2)
            end
            if it % print_iter == 1
                ΔA[i] = norm(A[i] - A_new[i], Inf)
            end
            # copy!(A[i], A_new[i])
            A[i] .= (1 - η) * A[i] + η * A_new[i]
        end
        if it % print_iter == 1
            obj = objective()
            loss = sum([sum(v) for (k, v) in obj])
            # @info "iteration $(it): ΔA= $(ΔA), L = $loss, obj = $(obj)"
            push!(trace, loss)
        end
    end
    S, A, trace
end
