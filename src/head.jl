using GZip

file_path = ARGS[1]

f = gzopen(joinpath(file_path))

for i=1:11
    print(readline(f))
end

close(f)