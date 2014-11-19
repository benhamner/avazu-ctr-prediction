using DataFrames
using GZip

typealias Feature  Uint64
typealias Features Vector{Feature}
typealias Weights  Vector{Float64}

type DataHash
    label::Float64
    features::Features
end

type DataPoint
    label::Float64
    features
end

type LogisticModel
    weights::Vector{Float64}
    counts::Vector{Float64}
    num_bits::Int
    alpha::Float64
end
initialize_model(num_bits, alpha) = LogisticModel(zeros(2^num_bits), zeros(2^num_bits), num_bits, alpha)

function hash_feature(model::LogisticModel, f::Any)
    (hash(f) >> (64 - model.num_bits))::Feature
end

function hash_features(model::LogisticModel, d::DataPoint)
    DataHash(d.label, [hash_feature(model, f) for f=d.features])
end

function hash_features(model::LogisticModel, d)
    [hash_feature(model, f) for f=d]
end

function predict(model::LogisticModel, feas::Features)
    pred = 0.0
    for f=feas
        pred += model.weights[f+1]
    end
    1 / (1. + exp(-max(min(pred, 20.), -20.)))
end

function learn!(model::LogisticModel, d::DataHash)
    pred = predict(model, d.features)
    update = (pred-d.label)*model.alpha
    for f = d.features
        model.weights[f+1] -= update / sqrt(model.counts[f+1]+1)
        model.counts[f+1] += 1.0
    end
    pred
end

function learn!(model::LogisticModel, d::DataPoint)
    learn!(model, hash_features(model, d))
end

function learn!(model::LogisticModel, df::DataFrame, features)
    loss = zeros(nrow(df))
    for i=1:nrow(df)
        pred = learn!(model, DataPoint(df[:click][i], [join([string(f),string(df[f][i])]) for f=features]))
        loss[i] = log_loss(df[:click][i], pred)
    end
    println("Mean Loss: ", mean(loss))
end

function predict_subset(df, features, model)
    probs = zeros(nrow(df), 1)
    for i=1:nrow(df)
        probs[i] = predict(model, hash_features(model, [join([string(f),string(df[f][i])]) for f=features]))
    end
    probs
end

function log_loss(y, p)
    p = max(min(p, 1. - 10e-12), 10e-12)
    (y == 1) ? -log(p) : -log(1. - p)
end

raw_data_path     = ARGS[1]
working_data_path = ARGS[2]

alpha    = .1
num_bits = 20
model = initialize_model(num_bits, alpha)

f = gzopen(joinpath(raw_data_path, "train.gz"), "r")

i = 0
header = readline(f)
raw    = [header]
#features = vcat([symbol(join(["I", i])) for i=1:13], [symbol(join(["C", i])) for i=1:26])
features = [symbol(x) for x=split(strip(header), ",")[3:end]]
println(features)

while !eof(f)
    i = i+1
    push!(raw, readline(f))
    if i%20_000==0 || eof(f)
        println(i/1e6, "m")
        df = readtable(IOBuffer(join(raw, "")))
        learn!(model, df, features)
        raw = [header]
    end
end

close(f)
println("Number of training lines ", i)

println("Reading test file ...")

f = gzopen(joinpath(raw_data_path, "test.gz"), "r")
header = readline(f)
raw    = [header]
i = 0
probs = Array(Float64, 0, 1)
ids   = Array(Int,     0)

while !eof(f)
    i = i+1
    push!(raw, readline(f))
    if i%100_000==0 || eof(f)
        println(i/1e6, "m")
        df = readtable(IOBuffer(join(raw, "")))
        probs = cat(1, probs, predict_subset(df, features, model))
        ids   = cat(1, ids, array(df[:id], 0))
        println("Len Probs", length(probs))
        println("Len Ids: ", length(ids))
        raw = [header]
    end
end

close(f)

submission = DataFrame(Id=ids, click=vec(probs))
writetable(joinpath(working_data_path, "submission.csv"), submission)