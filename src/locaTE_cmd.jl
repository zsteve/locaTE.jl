using locaTE
using OptimalTransport
using NPZ
using StatsBase
using SparseArrays
using ProgressMeter
using NearestNeighbors
using Graphs
using GraphSignals
using Printf
using ArgParse
using LinearAlgebra
using Discretizers
using NNlib
using Base.Filesystem
using CUDA

s = ArgParseSettings()
@add_arg_table s begin
    "X"
    help = "path to counts matrix, X"
    arg_type = String
    "X_rep"
    help = "path to dim-reduced representation of X"
    arg_type = String
    "P"
    help = "path to transition matrix"
    arg_type = String
    "R"
    help = "path to kernel matrix"
    arg_type = String
    "--tau"
    help = "power for transition matrix"
    arg_type = Int
    default = 1
    "--k_lap"
    help = "number of neighbours for Laplacian"
    arg_type = Int
    default = 15
    "--lambda1"
    arg_type = Float64
    default = 5.0
    "--lambda2"
    arg_type = Float64
    default = 0.01
    "--outdir"
    arg_type = String
    default = "./"
    "--suffix"
    arg_type = String
    default = ""
    "--cutoff"
    arg_type = Float64
    default = 0.0
    "--gpu"
    action = :store_true
    "--maxiter"
    arg_type = Int
    default = 1_000
end

args = parse_args(s)

function read_array(path)
    ext = splitext(path)[end]
    if ext == ".npy"
        # numpy array
        return npzread(path)
    elseif ext == ".csv"
        return CSV.read(path, Array)
    else
        @error "File extension $ext not supported. Please use npy or csv"
    end
end

@info "Reading input..."
X = read_array(args["X"])
# for Bayesian blocks, cutoff artifactually small counts 
X = relu.(X .- args["cutoff"])
X_rep = read_array(args["X_rep"])
P = read_array(args["P"])
Q = to_backward_kernel(P)
R = read_array(args["R"])

# construct kNN and Laplacian
kdtree = KDTree(X_rep')
idxs, dists = knn(kdtree, X_rep', args["k_lap"]);
A = spzeros(size(X_rep, 1), size(X_rep, 1));
for (i, j) in enumerate(idxs)
    A[i, j] .= 1.0
end
L = sparse(normalized_laplacian(max.(A, A'), Float64));

R_sp = sparse(R)
P_sp = sparse(P^args["tau"])
QT_sp = sparse((Q^args["tau"])')


@info "Estimating TE scores..."
TE =
    args["gpu"] ?
    estimate_TE_cu(
        X,
        1:size(X, 2),
        1:size(X, 2),
        Array(P_sp),
        Array(QT_sp),
        Array(R_sp);
        wclr = true,
        showprogress = true,
    ) :
    estimate_TE(
        X,
        1:size(X, 2),
        1:size(X, 2),
        P_sp,
        QT_sp,
        R_sp;
        wclr = true,
        showprogress = true,
    )

@info "Denoising..."
_cu = args["gpu"] ? x -> CUDA.cu(Array(x)) : x -> x
G = Array(
    fitsp(
        _cu(TE),
        _cu(L);
        λ1 = args["lambda1"],
        λ2 = args["lambda2"],
        maxiter = args["maxiter"],
    ),
)

npzwrite(string(args["outdir"], "G_$(args["suffix"]).npy"), G)
npzwrite(string(args["outdir"], "TE_$(args["suffix"]).npy"), TE)
npzwrite(string(args["outdir"], "L_$(args["suffix"]).npy"), Array(L))
