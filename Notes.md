# Hello Markdown
Modules support . indexing. I need to test if they are actually more memory efficient.
So I can call in the console julia <Profile Baselines.jl> stdout.txt to get the data 
I need to look into the difference between @allocated and @time

What is the difference between @time, @timed, and @timev

Profiling the basic vectorised operators
*, .*, .+, .-, ./, /, +, -

Findings thus far:
+= is faster than a = a + 
+ is a lot faster than .+
*= is a lot slower than a = a *
* is a lot faster than .*
^ does not work for matrixes 
.^= and a = a .^ are similar

Questions to answer:
How to store the time values for analysi
    Possibly Profile.retrieve
    Alternately it could be changing the profile(io argument)

Is range() or 1:DURATION faster?
Something interesting to try how local changed the time of assignment
It looks like assigning a matrix is much faster than assigning a vector using []
How does the precision of the assignment i.e. Float64, Float32 ec.t affect the assignment time
Ones and presumably zeros seem to be similar to the matrix assignment
How does the size of the array affect theassignment time.
Vector and matrix creation time seem similar.
How does this compare as the size increases.
Array vs MAtrix vs Vector 
Does the order of multiplication affet the time.
 I need to profile SparseMatrix (sparse(Matrix{Flaot64}(undef, 1, 1)))
I need to look into distributed sparse matrixes 
 @async is faster than @spawn is faster than @spawnat 

It takes ticktack 3 seconds to load
    timer = process_time(); import ticktack; timer = -=process_time();

I need to look into |> 
I am no longer using DistributedArrays
I am no longer looking into threads
SparseArrays may be next  

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