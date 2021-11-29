# Multi-threading, Multi-processing and Asynchronous Programming 
The _Threading_ and _Distributed_ libraries were trialled but abandoned
because they did not provide a speed boost in the scale of the model. 
Some of the more interesting functions to keep in mind are:
 - _@spawn_: Creates a worker process.
 - _@spawnat_: Creates a worker process at a specific thread.
 - _@task_: Creates a task that can be scheduled.
 - _@async_: Creates and schedules a task for the os.

# Matrix and Linear Algebra
The _DistributedArrays_, _SparseArrays_ and _Diagonal_ libraries were
trialled. 
 - _sparse()_: A good function, but not efficient for diagonals.
    - Awaiting a descision for the transfer operator and flux matrix.
 - _Diagonal()_: Extremely efficient for diagonal arrays.
 - _Distributed()_: Not nessecary for the scale of the model. Splits the
 array in RAM for efficiency.

# Profiling
I need to try __BenchmarkTools.jl__, although _Profiling_ is working for
the smaller stuff.

# Name Spaces
I can use the following declaration pattern to fully static type in julia

    local a::Int16 = 1;
