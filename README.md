These are conceptual ideas for de-entropy in compression data.

Duckomp-v4: Hunts repeated substrings that will be outside of the Zstd window of reference. 
Then hunts polynomial equations that round to their integer offsets (4 points, more takes too long as of now and would lower equation compression). 
The equations are written in hyper-shorthand that is decoded when run. 
As of my testing it has a 100% success rate, but fails to compress data enough for useful returns.

Duckomp-v5: Reads offsets (which are meant to be substrings) and hunts for seeds that match those offsets using GPU.
The idea here is that substrings outside of a Zstd window are ALWAYS held in a single seed, computing it is the only constraint.
As of my testing, this does work on smaller data sets, but the time increase on larger offsets is exponential.
