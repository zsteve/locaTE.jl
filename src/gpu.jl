function accum_joint_probs_dense!(gamma, coupling, ids, offset_x, N_x, offset_y, N_y)
    # for full block iteration, set offset_x = offset_y = 0 and N_x = size(gamma, 1), etc. 
    index_i = (blockIdx().x-1)*blockDim().x + threadIdx().x
    index_j = (blockIdx().y-1)*blockDim().y + threadIdx().y
    index_k = (blockIdx().z-1)*blockDim().z + threadIdx().z
    # stride_i, stride_j, stride_k = blockDim().x, blockDim().y, blockDim().z
    stride_i = gridDim().x * blockDim().x
    stride_j = gridDim().y * blockDim().y
    stride_k = gridDim().z * blockDim().z
    # for i = index_i:stride_i:(size(gamma, 1)*size(gamma, 2))
    for i = index_i:stride_i:N_x*N_y
        # i0 = ((i-1) % size(gamma, 1)) + 1
        # i1 = ((i-1) ÷ size(gamma, 1)) + 1
        i0_abs = ((i-1) % N_x) + 1 + offset_x
        i1_abs = ((i-1) ÷ N_x) + 1 + offset_y
        i0 = ((i-1) % N_x) + 1
        i1 = ((i-1) ÷ N_x) + 1
        for j = index_j:stride_j:size(coupling, 1)
            for k = index_k:stride_k:size(coupling, 2)
                @inbounds CUDA.@atomic gamma[i0, i1, ids[j, i0_abs], ids[k, i1_abs], ids[j, i1_abs]] += coupling[j, k]
            end
        end
    end
    return nothing
end

function accum_joint_probs_sparse!(gamma, coupling_I, coupling_J, coupling_V, ids, offset_x, N_x, offset_y, N_y)
    index_i = (blockIdx().x-1)*blockDim().x + threadIdx().x
    index_j = (blockIdx().y-1)*blockDim().y + threadIdx().y
    index_k = (blockIdx().z-1)*blockDim().z + threadIdx().z
    stride_i = gridDim().x * blockDim().x
    stride_j = gridDim().y * blockDim().y
    stride_k = gridDim().z * blockDim().z
    for i = index_i:stride_i:N_x
        for j = index_j:stride_j:N_y
            i_abs = i + offset_x
            j_abs = j + offset_y
            for k = index_k:stride_k:length(coupling_I)
                @inbounds CUDA.@atomic gamma[i, j, ids[coupling_I[k], i_abs], ids[coupling_J[k], j_abs], ids[coupling_I[k], j_abs]] += coupling_V[k] 
            end
        end
    end
    return nothing
end

function getcoupling_dense(i, P, QT, R)
    # full coupling
    pi = R[i, :]
    QT * (reshape(pi, :, 1) .* P)
end

function getcoupling_dense_trimmed(i, P, QT, R)
    # full coupling but remove empty rows/cols
    pi = R[i, :]
    coupling = QT * (reshape(pi, :, 1) .* P)
    row_idxs = vec(sum(coupling; dims = 2) .> 0)
    col_idxs = vec(sum(coupling; dims = 1) .> 0)
    return coupling[row_idxs, :][:, col_idxs]
end

function getcoupling_sparse(i, P, QT, R)
    # return list of (i, j, v) for sparse coupling representation
    pi = R[i, :]
    coupling = QT * (reshape(pi, :, 1) .* P)
    findnz(sparse(coupling)) 
end

function conditional_mutual_information(joint_probs)
    H_xz = dropdims(mapreduce(xlogx, +, sum(joint_probs; dims = 4); dims = (3, 5)))
    H_yz = dropdims(mapreduce(xlogx, +, sum(joint_probs; dims = 3); dims = (4, 5)))
    H_xyz = dropdims(mapreduce(xlogx, +, joint_probs; dims = (3, 4, 5)))
    H_z = dropdims(mapreduce(xlogx, +, sum(joint_probs; dims = (3, 4)); dims = 5))
    -H_xz - H_yz + H_xyz + H_z
end

function get_joint_cache(N_genes, discret_max_size)
    CUDA.fill(0f0, N_genes, N_genes, fill(discret_max_size, 3)...); 
end

function get_joint_cache(N_x, N_y, discret_max_size)
    CUDA.fill(0f0, N_x, N_y, fill(discret_max_size, 3)...); 
end

function get_MI!(mi_all, joint_cache, coupling_I, coupling_J, coupling_V, N_genes, ids; threads=(8,8,8), blocks=128, offset_x = nothing, N_x = nothing, offset_y = nothing, N_y = nothing)
    CUDA.fill!(joint_cache, 0f0)
    offset_x = (offset_x === nothing) ? 0 : offset_x
    offset_y = (offset_y === nothing) ? 0 : offset_y
    N_x = (N_x === nothing) ? size(joint_cache, 1) : N_x
    N_y = (N_y === nothing) ? size(joint_cache, 2) : N_y
    CUDA.@sync begin
        @cuda threads=threads blocks=blocks accum_joint_probs_sparse!(joint_cache, coupling_I, coupling_J, coupling_V, ids, offset_x, N_x, offset_y, N_y)
    end
    idx1=(1+offset_x):(N_x+offset_x)
    idx2=(1+offset_y):(N_y+offset_y)
    copy!(view(mi_all, idx1, idx2), conditional_mutual_information(joint_cache[1:N_x, 1:N_y, :, :]))
end

function get_MI!(mi_all, joint_cache, coupling, N_genes, ids; threads=(8,8,8), blocks=128, offset_x = nothing, N_x = nothing, offset_y = nothing, N_y = nothing)
    CUDA.fill!(joint_cache, 0f0)
    offset_x = (offset_x === nothing) ? 0 : offset_x
    offset_y = (offset_y === nothing) ? 0 : offset_y
    N_x = (N_x === nothing) ? size(joint_cache, 1) : N_x
    N_y = (N_y === nothing) ? size(joint_cache, 2) : N_y
    CUDA.@sync begin
        @cuda threads=threads blocks=blocks accum_joint_probs_dense!(joint_cache, coupling, ids, offset_x, N_x, offset_y, N_y)
    end
    idx1=(1+offset_x):(N_x+offset_x)
    idx2=(1+offset_y):(N_y+offset_y)
    copy!(view(mi_all, idx1, idx2), conditional_mutual_information(joint_cache[1:N_x, 1:N_y, :, :]))
end

function getblocks(N_genes, blocks_x, blocks_y)
    quot_x, rem_x = N_genes ÷ blocks_x, N_genes % blocks_x
    N_x = fill(quot_x, blocks_x)
    if rem_x > 0
        push!(N_x, rem_x)
    end
    offset_x = pushfirst!(cumsum(N_x)[1:end-1], 0)
    # now for y
    quot_y, rem_y = N_genes ÷ blocks_y, N_genes % blocks_y
    N_y = fill(quot_y, blocks_y)
    if rem_y > 0
        push!(N_y, rem_y)
    end
    offset_y = pushfirst!(cumsum(N_y)[1:end-1], 0)
    return zip(Base.Iterators.product(N_x, N_y), Base.Iterators.product(offset_x, offset_y))
end
