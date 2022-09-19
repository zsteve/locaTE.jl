function accum_joint_probs_dense!(gamma, coupling, ids)
    index_i = (blockIdx().x-1)*blockDim().x + threadIdx().x
    index_j = (blockIdx().y-1)*blockDim().y + threadIdx().y
    index_k = (blockIdx().z-1)*blockDim().z + threadIdx().z
    # stride_i, stride_j, stride_k = blockDim().x, blockDim().y, blockDim().z
    stride_i = gridDim().x * blockDim().x
    stride_j = gridDim().y * blockDim().y
    stride_k = gridDim().z * blockDim().z
    for i = index_i:stride_i:(size(gamma, 1)*size(gamma, 2))
        i0 = ((i-1) % size(gamma, 1)) + 1
        i1 = ((i-1) ÷ size(gamma, 1)) + 1
        for j = index_j:stride_j:size(coupling, 1)
            for k = index_k:stride_k:size(coupling, 2)
                @inbounds CUDA.@atomic gamma[i0, i1, ids[j, i0], ids[k, i1], ids[j, i1]] += coupling[j, k]
            end
        end
    end
    return nothing
end

function accum_joint_probs_sparse!(gamma, coupling_I, coupling_J, coupling_V, ids)
    index_i = (blockIdx().x-1)*blockDim().x + threadIdx().x
    index_j = (blockIdx().y-1)*blockDim().y + threadIdx().y
    index_k = (blockIdx().z-1)*blockDim().z + threadIdx().z
    stride_i = gridDim().x * blockDim().x
    stride_j = gridDim().y * blockDim().y
    stride_k = gridDim().z * blockDim().z
    for i = index_i:stride_i:size(gamma, 1)
        for j = index_j:stride_j:size(gamma, 2)
            for k = index_k:stride_k:length(coupling_I)
                @inbounds CUDA.@atomic gamma[i, j, ids[coupling_I[k], i], ids[coupling_J[k], j], ids[coupling_I[k], j]] += coupling_V[k] 
            end
        end
    end
    return nothing
end

function getcoupling_dense(i, P, QT, R)
    pi = R[i, :]
    QT * (reshape(pi, :, 1) .* P)
end

function getcoupling_dense_trimmed(i, P, QT, R)
    pi = R[i, :]
    coupling = QT * (reshape(pi, :, 1) .* P)
    row_idxs = vec(sum(coupling; dims = 2) .> 0)
    col_idxs = vec(sum(coupling; dims = 1) .> 0)
    return coupling[row_idxs, :][:, col_idxs]
end

function getcoupling_sparse(i, P, QT, R)
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

function get_MI!(mi_all, joint_cache, coupling_I, coupling_J, coupling_V, N_genes, ids; threads=(8,8,8), blocks=128)
    CUDA.fill!(joint_cache, 0f0)
    CUDA.@sync begin
        @cuda threads=threads blocks=blocks accum_joint_probs_sparse!(joint_cache, coupling_I, coupling_J, coupling_V, ids)
    end
    copy!(mi_all, conditional_mutual_information(joint_cache))
end

function get_MI!(mi_all, joint_cache, coupling, N_genes, ids; threads=(8,8,8), blocks=128)
    CUDA.fill!(joint_cache, 0f0)
    CUDA.@sync begin
        @cuda threads=threads blocks=blocks accum_joint_probs_dense!(joint_cache, coupling, ids)
    end
    copy!(mi_all, conditional_mutual_information(joint_cache))
end
