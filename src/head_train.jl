using GZip

raw_data_path = ARGS[1]

f = gzopen(joinpath(raw_data_path, "train_rev2.gz"))

for i=1:11
    print(readline(f))
end

close(f)