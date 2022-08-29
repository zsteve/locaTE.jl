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

function fitnmf(G, L, K1, K2, k; α = 5, λ1 = 1e-2, λ2 = 1e-2, μ1 = 1e-2, μ2 = 1e-2, iter = 500, print_iter = 50, initialize = :nndsvd)
    # use multiplicative updates
    U = similar(G, size(G, 1), k)
    U_new = similar(U)
    V = similar(G, size(G, 2), k)
    V_new = similar(V)
    # initialize
    @info "Initializing as $(initialize)"
    if initialize === :rand
        U .= rand(size(U))
        V .= rand(size(V))
    elseif initialize === :nndsvd
        U_init, V_init = NMF.nndsvd(G, k)
        copy!(U, U_init)
        copy!(V, V_init')
    end
    for it = 1:iter
        # U_new .= U .* ((G*V) ./ (U*V'*V + λ1*(L*U) .+ λ2))
        # V_new .= V .* ((G'*U) ./ (V*U'*U .+ μ))
        U_new .= U .* ((G*V) ./ (U*V'*V + α*(L*U*V'*V) + λ1*(K1*U) .+ λ2))
        V_new .= V .* ((G'*U) ./ (V*U'*U + α*(V*U'*L*U) + μ1*(K2*V) .+ μ2))
        η = 0.5
        U .= (1-η)U + η*U_new
        V .= (1-η)V + η*V_new
        if (it % print_iter == 1)
            ΔU, ΔV = norm(U - U_new, Inf), norm(V - V_new, Inf)
            @info "iteration $(it), (ΔU, ΔV) = $((ΔU, ΔV))"
        end
    end
    relu.(U), relu.(V)
end
