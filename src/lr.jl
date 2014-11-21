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
header_fea = [x*"-" for x=split(strip(header), ",")]
println(header_fea)
loss = 0

while !eof(f)
    i = i+1
    row = split(readline(f), ",")
    fea = [header_fea[i]*row[i] for i=3:length(row)]
    click = int(row[2])
    pred = learn!(model, DataPoint(click, fea))
    loss += log_loss(click, pred)
    if i%100_000==0 || eof(f)
        println(i/1e6, "m", " Loss ", loss/100_000)
        loss=0
    end
end

close(f)
println("Number of training lines ", i)

println("Reading test file ...")

f = gzopen(joinpath(raw_data_path, "test.gz"), "r")
header = readline(f)
raw    = [header]
i = 0
probs = Array(Float64, 0)
ids   = Array(ASCIIString, 0)

while !eof(f)
    i = i+1
    row = split(readline(f), ",")
    fea = [header_fea[i+1]*row[i] for i=2:length(row)]
    
    push!(ids, ASCIIString(row[1]))
    push!(probs, predict(model, hash_features(model, fea)))

    if i%100_000==0 || eof(f)
        println(i/1e6, "m")
    end
end

close(f)

submission = DataFrame(id=[string(x) for x=ids], click=vec(probs))
writetable(joinpath(working_data_path, "submission.csv"), submission)