using locaTE
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

s = ArgParseSettings(
    description = "locaTE-cmd: utility for running locaTE workflow from the command-line. Input matrices can be supplied as .npy (binary) or .csv (text).",
)
@add_arg_table s begin
    "X"
    help = "Path to counts matrix X"
    arg_type = String
    required = true
    "X_rep"
    help = "Path to dimensionality-reduced representation of X. From this, the kNN graph will be constructed."
    arg_type = String
    required = true
    "P"
    help = "Path to transition matrix P encoding dynamics."
    arg_type = String
    required = true
    "R"
    help = "Path to kernel matrix R encoding neighbourhood information."
    arg_type = String
    required = true
    "--tau"
    help = "Power for transition matrix."
    arg_type = Int
    default = 1
    "--k_lap"
    help = "Number of neighbours for Laplacian."
    arg_type = Int
    default = 15
    "--lambda1"
    help = "lambda1 (λ1), strength of Laplacian regularization."
    arg_type = Float64
    default = 5.0
    "--lambda2"
    help = "lambda2 (λ2), strength of Lasso regularization."
    arg_type = Float64
    default = 0.01
    "--outdir"
    help = "Output directory"
    arg_type = String
    default = "./"
    "--suffix"
    help = "Suffix to append to output files."
    arg_type = String
    default = ""
    "--cutoff"
    help = "Cutoff below which expression values will be set to zero. Can be helpful for expression value binning in some datasets with artifactually small counts."
    arg_type = Float64
    default = 0.0
    "--gpu"
    help = "GPU acceleration, recommended for large datasets when available."
    action = :store_true
    "--maxiter"
    help = "Maximum iterations for denoising regression."
    arg_type = Int
    default = 1_000
end

args = parse_args(s)
args["suffix"] = args["suffix"] == "" ? "" : string("_", args["suffix"])

function read_array(path)
    ext = splitext(path)[end]
    if ext == ".npy"
        # numpy array
        return npzread(path)
    elseif ext == ".csv"
        # text format (csv)
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

npzwrite(joinpath(args["outdir"], "G$(args["suffix"]).npy"), G)
npzwrite(joinpath(args["outdir"], "TE$(args["suffix"]).npy"), TE)
npzwrite(joinpath(args["outdir"], "L$(args["suffix"]).npy"), Array(L))
