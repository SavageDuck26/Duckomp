These are conceptual ideas for de-entropy in compression data.

**Duckomp-v4:** Hunts repeated substrings that will be outside of the Zstd window of reference. 
Then hunts polynomial equations that round to their integer offsets (4 points, more takes too long as of now and would lower equation compression). 
The equations are written in hyper-shorthand that is decoded when run. 
As of my testing it has a 100% success rate, but fails to compress data enough for useful returns.

**Duckomp-v5:** Reads offsets (which are meant to be substrings) and hunts for seeds that match those offsets using GPU.
The idea here is that substrings outside of a Zstd window are ALWAYS held in a single seed, computing it is the only constraint.
As of my testing, this does work on smaller data sets, but the time increase on larger offsets is exponential.

**Duckomp-v6:** Creates a "phonebook" of every possible offset given the data. 
The next step is to make a file to binary search through the offset permutations to find the right one, essentially v5 but fast.
v6 is meant to show the process and under-the-hood view of how it works. Such as:

Offsets: 3, 4, 1, 4, 1. (Which would come from 'random' data that would be much larger in practice)
Perm 0: [1,1,1,1,1]
Perm 40: [1,2,2,2,2]
Perm 41: [1,2,2,2,3]
Perm 42: [1,2,2,3,1]
Perm 106: [2,1,3,3,2]
Perm 107: [2,1,3,3,3]
Perm 1022: [4,4,4,4,3]
Perm 1023: [4,4,4,4,4]

**Duckomp-v7:** Reads offsets and searches through them to find the "Seed".
That seed is an integer, which then gets 'solvered' down to a smaller equation (Not optimized yet).
This works very fast for massive data sets, but needs to be tested on data for Zstd, one can hope.
