import random
import sys



def default_sample():
    offset_low = 1
    offset_high = 4
    count = 18
    
    return (offset_low, offset_high, count)

def calc_rand_offset(input):
    offset_low, offset_high, count = input

    output = []
    prev = None

    for i in range(count):
        n = random.randint(offset_low, offset_high)
        while n == prev:
            n = random.randint(offset_low, offset_high)
        output.append(n)
        prev = n

    return output

# Big structured sample: alternating tight clusters and wide spreads, so the
# gaps are non-stationary — the case DP chunking is meant to win on.
def big_sample(segments=20, steps=100):
    out = []
    for i in range(segments):
        if i % 2 == 0:
            out.append((steps, 1, 3))        # tight cluster
        else:
            out.append((steps, 300, 900))    # wide spread
    return out

def calc_segmented_offsets(segments):
    pos = 1
    output = [pos]
    for count, low, high in segments:
        for _ in range(count):
            pos += random.randint(low, high)
            output.append(pos)
    return output

def main():
    path = "sample.txt"
        
    if len(sys.argv) > 1 and sys.argv[1] == "big":
        segments = int(sys.argv[2]) if len(sys.argv) > 2 else 20
        steps = int(sys.argv[3]) if len(sys.argv) > 3 else 100
        output = calc_segmented_offsets(big_sample(segments, steps))
    else:
        output = calc_rand_offset(default_sample())
    
    with open(path, "w") as file:
        for offset in output:
            file.write(str(offset) + "\n")
        


if __name__ == "__main__":
    main()