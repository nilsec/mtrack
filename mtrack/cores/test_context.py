import numpy as np
from fractions import gcd 

def lcm(denominators):
    return reduce(lambda a,b: a*b // gcd(a,b), denominators)

def get_conflict_free_sets2d(grid, nbsize, dim=2, pad=True):
    grid_size = np.shape(grid)[0] * np.shape(grid)[1]

    if pad:
        padded_dim = lcm([np.shape(grid)[0], np.shape(grid)[1], nbsize])
        padded_grid = -1 * np.ones([padded_dim] * 2, dtype=int)
    
        padded_grid[:grid.shape[0], :grid.shape[1]] = grid
        grid = padded_grid

    n_sets = nbsize**dim
    cf_lists = []

    for n in range(n_sets):
        cf = []
        p0 = np.array([n % nbsize, (n/nbsize)])
        cf = [p0 + np.array([nbsize * i, nbsize * j]) for i in range(np.shape(grid)[0]/nbsize)\
                                                      for j in range(np.shape(grid)[1]/nbsize)]
        cf_lists.append([m for m in cf if grid[m[0], m[1]] != -1])

    test_list = []
    for c in cf_lists:
        print "cf:"
        for p in c:
            test_list.append(grid[p[0], p[1]])
            print grid[p[0], p[1]]
        print "\n"

    print sorted(test_list)
    print np.all(np.array(sorted(test_list)) == np.arange(grid_size))


def get_cf_3d(cube, nbsize, dim=3):
    n_sets = nbsize**dim
    cf_lists = []

    for n in range(n_sets):
        cf = []
        p0 = np.array([n%nbsize, (n/nbsize), n/nbsize])
        print p0


if __name__ == "__main__":
    """
    cube = np.arange(36*6).reshape(6,6,6)
    get_cf_3d(cube, 2)
    """

    grid = np.arange(7*5).reshape(7,5)
    nbsize = 8
    get_conflict_free_sets2d(grid, nbsize, pad=True)
