def check_overlap(x, y):
    xs = set(range(min(x), max(x)))
    ys = set(range(min(y), max(y)))

    if xs.intersection(ys):
        return True
    else:
        return False

if __name__ == "__main__":
    print check_overlap([1,10], [-1,-10])
    
