prox_l1(x; λ = 1.0) = sign(x)*relu(abs(x) - λ)

function fitsp(G::AbstractMatrix, L::AbstractMatrix, α; ρ = 0.05, λ1 = 25.0, λ2 = 0.075, maxiter = 2500)
    # scaling factors
    L_scaled = L # sqrt.(α) * L * sqrt.(α)
    X = G;
    Z = G;
    W = zero(G);
    ΔX, ΔZ, ΔW = 0, 0, 0
    @showprogress for iter = 1:maxiter
        # X_new = ((1+ρ)I + λ1*L) \ (G - ρ*(W-Z));
        X_new = (α + λ1*L_scaled + ρ*I) \ (α*G + ρ*(Z-W)); 
        # Z_new = prox_l1.(X_new+W; λ = λ2/ρ);
        Z_new = hcat([prox_l1.(X_new[i, :]+W[i, :]; λ = λ2*diag(α)[i]/ρ) for i = 1:size(L, 1)]...)';
        W_new = W + X_new - Z_new
        ΔX, ΔZ, ΔW = norm(X-X_new, Inf), norm(Z-Z_new, Inf), norm(W-W_new, Inf)
        X = X_new; Z = Z_new; W = W_new
    end
    @info "ΔX = $(ΔX), ΔZ = $(ΔZ), ΔW = $(ΔW)"
    @info "tr(X'LX) = $(tr(X'*L_scaled*X)), 0.5|X-G|^2 = $(0.5*norm(X-G)), |X|1 = $(norm(X, 1))"
    Z
end

function fitsp(G::AbstractMatrix, L::AbstractMatrix; ρ = 0.05, λ1 = 25.0, λ2 = 0.075, maxiter = 2500)
    # scaling factors
    X = G;
    Z = G;
    W = zero(G);
    ΔX, ΔZ, ΔW = 0, 0, 0
    @showprogress for iter = 1:maxiter
        X_new = ((1+ρ)I + λ1*L) \ (G - ρ*(W-Z));
        Z_new = prox_l1.(X_new+W; λ = λ2/ρ);
        W_new = W + X_new - Z_new
        ΔX, ΔZ, ΔW = norm(X-X_new, Inf), norm(Z-Z_new, Inf), norm(W-W_new, Inf)
        X = X_new; Z = Z_new; W = W_new
    end
    @info "ΔX = $(ΔX), ΔZ = $(ΔZ), ΔW = $(ΔW)"
    @info "tr(X'LX) = $(tr(X'*L*X)), 0.5|X-G|^2 = $(0.5*norm(X-G)), |X|1 = $(norm(X, 1))"
    Z
end


function fitsp_mean(G; ρ = 0.01, λ = 0.001, maxiter = 100)
    x = similar(G, size(G, 1))
    x_new = similar(x)
    z = similar(G, size(G, 1))
    z_new = similar(z)
    w = similar(G, size(G, 1))
    w_new = similar(w)
    fill!(x, 0)
    fill!(z, 0)
    fill!(w, 0)
    Δx, Δz, Δw = 0, 0, 0
    @showprogress for iter = 1:maxiter
        x_new =  (mean(G; dims = 2) + ρ*(z-w))/(1+ρ)
        z_new = prox_l1.(x_new + w; λ = λ/ρ)
        w_new = w + x_new - z_new
        Δx, Δz, Δw = norm(x-x_new, Inf), norm(z-z_new, Inf), norm(w-w_new, Inf)
        copy!(z, x_new)
        copy!(x, x_new)
        copy!(w, w_new)
    end
    @info "Δx = $(Δx), Δz = $(Δz), Δw = $(Δw)"
    x
end

function fitnmf(G, L_all, L, H, k; α = 1e-2, β = 0, λ = [1e-2, 1e-2], μ = [1e-2, 1e-2], iter = 500, print_iter = 50, initialize = :nndsvd, δ = 1e-5, dictionary = false, η = 1.0)
    # use multiplicative updates
    U = similar(G, size(G, 1), k)
    U_new = similar(U)
    V = similar(G, size(G, 2), k)
    V_new = similar(V)
    #
    D = Diagonal(diag(L))
    A = D - L 
    D1 = Diagonal(diag(L_all[1]))
    D2 = Diagonal(diag(L_all[2]))
    A1 = D1 - L_all[1]
    A2 = D2 - L_all[2]
    # initialize
    @info "Initializing NMF decomposition with $(initialize)"
    if initialize === :rand
        copy!(U, rand(size(U)))
        copy!(V, rand(size(V)))
    elseif initialize === :nndsvd
        U_init, V_init = NMF.nndsvd(G, k)
        copy!(U, U_init)
        copy!(V, V_init')
    elseif initialize === :nmf
        tmp = nnmf(G, k; alg = :multmse, maxiter = 2)
        copy!(U, tmp.W)
        copy!(V, tmp.H')
    end
    if dictionary
        V *= mean(sum(U; dims = 2))
        U ./= sum(U; dims = 2)
    end
    trace = []
    for it = 1:iter
        function objective()
            Dict(:df => norm(U*V' - G, 2)^2 / 2,
                 :smooth => α/2*tr(V*U'*L*(U*V')),
                 :affine => -β*tr(H'*U*V'), 
                 :smooth_U => λ[1]/2*tr(U'*L_all[1]*U),
                 :sp_U => λ[2]*norm(U, 1),
                 :smooth_V => μ[1]/2*tr(V'*L_all[2]*V),
                 :sp_V => μ[2]*norm(V, 1))
        end
        U_new .= relu.(U .* (G*V + λ[1]*A1*U + α*A*U*V'*V + β*H*V)) ./ (α*D*U*V'*V + U*V'*V + λ[1]*(D1*U) .+ λ[2] .+ δ)
        if dictionary
            U_new ./= sum(U_new; dims = 2)
        end
        V_new .= relu.(V .* (G'*U + μ[1]*A2*V + α*V*U'*A*U + β*H'*U)) ./ (α*V*U'*D*U + V*U'*U + μ[1]*(D2*V) .+ μ[2] .+ δ)
        if (it % print_iter == 0)
            ΔU, ΔV = norm(U - U_new, Inf), norm(V - V_new, Inf)
            obj = objective()
            @info obj
            l = sum([v for (k, v) in obj])
            @info "iteration $(it), (ΔU, ΔV) = $((ΔU, ΔV)), L = $(l), objs = $(obj)"
            push!(trace, l)
        end
        # copy!(U, U_new)
        # copy!(V, V_new)
        U .= (1-η)*U + η*U_new
        V .= (1-η)*V + η*V_new
    end
    U, V, trace
end

function fitntf(G, L, L_g, λ, μ, α, k; iter = 250, print_iter = 50, dictionary = false, δ = 1e-5, η = 1.0)
    # factor matrices
    A = [similar(G, size(G, i), k) for i = 1:length(size(G))]
    A_new = [similar(a) for a in A]
    # core tensor
    S = zeros([k for _ = 1:length(size(G))]...)
    for i = 1:k
        S[i,i,i] = 1.0
    end
    # initialization 
    factor_cp = tl_decomp.non_negative_parafac(G, rank = k, init = "svd", n_iter_max = 1, random_state = 0)
    for i = 1:length(A)
        copy!(A[i], factor_cp.factors[i])
    end
    # normalize factor A^{(1)} if we need dictionary interpretation
    if dictionary
        A[1] ./= sum(A[1]; dims = 2)
    end
    # 
    D = [Diagonal(diag(l)) for l in L]
    W = [d - l  for (d, l) in zip(D, L)]
    D_g = Diagonal(diag(L_g))
    W_g = D_g - L_g
    ΔA = zeros(length(A))
    trace = []
    for it = 1:iter
        function objective()
            X = ttm(S, A, 1:length(A))
            X1 = tenmat(X, 1)
            Dict(:df => norm(X - G, 2)^2 / 2,
                 :smooth => [ λ[i]/2 * tr(A[i]'*L[i]*A[i]) for i = 1:length(A) ],
                 :sparse => [ μ[i] * norm(A[i], 1) for i = 1:length(A) ],
                 :smooth_mode1 => α/2*tr(X1'*L_g*X1)
                    )
        end
        for i = 1:length(A)
            inds = [j for j in range(length=length(A)) if j != i]
            Bi = tenmat(ttm(S, [Matrix(A[j]) for j in inds], inds), i)
            if i == 1
                A_new[i] .= relu.(A[i] .* (tenmat(G, i) * Bi' + λ[i]*W[i]*A[i] + α*W_g*A[i]*Bi*Bi') ./ (A[i]*Bi*Bi' + λ[i]*D[i]*A[i] + α*D_g*A[i]*Bi*Bi' .+ μ[i] .+ δ))
            else
                Btilde_plus = tenmat(ttm(S, [Matrix((j == 1 ? D_g : I) * A[j]) for j in inds], inds), i)
                Btilde_minus = tenmat(ttm(S, [Matrix((j == 1 ? W_g : I) * A[j]) for j in inds], inds), i)
                C_plus = (Btilde_plus*Bi' + Bi*Btilde_plus')/2
                C_minus = (Btilde_minus*Bi' + Bi*Btilde_minus')/2
                A_new[i] .= relu.(A[i] .* (tenmat(G, i) * Bi' + λ[i]*W[i]*A[i] + α*A[i]*C_minus) ./ (A[i]*Bi*Bi' + λ[i]*D[i]*A[i] + α*A[i]*C_plus .+ μ[i] .+ δ))
            end
            # normalize first factor
            if dictionary && (i == 1)
                A_new[i] ./= sum(A_new[i]; dims = 2)
            end
            if it % print_iter == 1
                ΔA[i] = norm(A[i] - A_new[i], Inf)
            end
            # copy!(A[i], A_new[i])
            A[i] .= (1-η)*A[i] + η*A_new[i]
        end
        if it % print_iter == 1
            obj = objective()
            loss = sum([sum(v) for (k, v) in obj])
            @info "iteration $(it): ΔA= $(ΔA), L = $loss, obj = $(obj)"
            push!(trace, loss)
        end
    end
    S, A, trace
end
