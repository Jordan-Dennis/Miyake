using Profile;

global const SAMPLE = 1e3;  # The number of samples for the basic operators 

# Profiling the basic assignment operators
# =, +=, -=, *=, /= 
@allocated @time for i in 1:SAMPLE; a = 1; end;
@allocated @time for i in 1:SAMPLE; a += 1; end;
@allocated @time for i in 1:SAMPLE; a -= 1; end;
@allocated @time for i in 1:SAMPLE; a *= 1; end;
@allocated @time for i in 1:SAMPLE; a /= 1; end;

module AccessTest
    const CONSTANT = "A constant";  # Profiling the const declaration 
    local variable = "A variable";  # Profiling the local declaration 
    function _function()
        return variable + CONSTANT;
    end
    variable = _function()
end
# Modules support . indexing. I need to test if they are actually more memory efficient.
# So I can call in the console julia <Profile Baselines.jl> stdout.txt to get the data 
# I need to look into how to create my own macros 
# I also need to investigate multithreading
# I need to look into the difference between @allocated and @time

# What is the difference between @time, @timed, and @timev

# Profiling the basic vectorised operators
# *, .*, .+, .-, ./, /, +, -

# Findings thus far:
# += is faster than a = a + 
# + is a lot faster than .+
# *= is a lot slower than a = a *
# * is a lot faster than .*
# ^ does not work for matrixes 
# .^= and a = a .^ are similar

# Questions to answer:
# How to store the time values for analysi
    # Possibly Profile.retrieve
    # Alternately it could be changing the profile(io argument )

# Is range() or 1:DURATION faster?
# Something interesting to try how local changed the time of assignment
# It looks like assigning a matrix is much faster than assigning a vector using []
# How does the precision of the assignment i.e. Float64, Float32 ec.t affect the assignment time
# Ones and presumably zeros seem to be similar to the matrix assignment
# How does the size of the array affect theassignment time.
# Vector and matrix creation time seem similar.
# How does this compare as the size increases.
# Array vs MAtrix vs Vector 
# Does the order of multiplication affet the time.

# I need to profile SparseMatrix (sparse(Matrix{Flaot64}(undef, 1, 1)))