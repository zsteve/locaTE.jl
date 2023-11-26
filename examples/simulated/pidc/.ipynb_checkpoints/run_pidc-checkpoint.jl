using Pkg
Pkg.activate("pidc_env")
Pkg.add(["NetworkInference", "DataFrames", "CSV", "NPZ", "StatsBase", "LinearAlgebra", "NNlib"])
using NetworkInference
using DataFrames
using CSV
using NPZ
using StatsBase
using LinearAlgebra
using NNlib

idx = Colon()
X = npzread("X.npy")[:, idx]
try
    global genes = CSV.read("genes.txt", Array)[:, 2][idx];
catch e
    global genes = ["gene_$i" for i = 1:size(X, 2)];
end
df = DataFrame(X', :auto)
# df = DataFrame(X')
insertcols!(df, 1, :gene => genes)
CSV.write("X.csv", df, delim = "\t")

n = infer_network("X.csv", PIDCNetworkInference(), discretizer = "bayesian_blocks");
w = [e.weight for e in n.edges]
e = [(e.nodes[1].label, e.nodes[2].label) for e in n.edges];
A = zeros(size(X, 2), size(X, 2))
for (w_, e_) in zip(w, e)
    A[findfirst(x -> x == e_[1], genes)[1], 
      findfirst(x -> x == e_[2], genes)[1]] = w_
end
A = A + A';
npzwrite("G_pidc.npy", A)
