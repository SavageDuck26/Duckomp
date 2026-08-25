import random



def default_sample():
    offset_low = 1
    offset_high = 4
    count = 5
    
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

def main():
    path = "sample.txt"
        
    sample = default_sample()
    output = calc_rand_offset(sample)
    
    with open(path, "w") as file:
        for offset in output:
            file.write(str(offset) + "\n")
        


if __name__ == "__main__":
    main()