import numpy as np
from fractions import gcd 
import time

def lcm(denominators):
    return reduce(lambda a,b: a*b // gcd(a,b), denominators)

def get_conflict_free_sets2d(grid, nbsize, dim=2, pad=True, virtual=True):
    grid_size = np.shape(grid)[0] * np.shape(grid)[1]

    if pad:
        padded_dim = lcm([np.shape(grid)[0], np.shape(grid)[1], nbsize])
        if not virtual:
            padded_grid = -1 * np.ones([padded_dim] * 2, dtype=int)
            padded_grid[:grid.shape[0], :grid.shape[1]] = grid
            grid = padded_grid

    n_sets = nbsize**dim
    cf_lists = []

    for n in range(n_sets):
        cf = []
        p0 = np.array([n % nbsize, (n/nbsize)])
        cf = [p0 + np.array([nbsize * i, nbsize * j]) for i in range(padded_dim/nbsize)\
                                                      for j in range(padded_dim/nbsize)]
        
        if virtual:
            m_list = [] 
            for m in cf:
                try:
                    m_list.append(m)
                except IndexError:
                    pass
            if m_list:
                cf_lists.append(m_list)
        else:
            cf_lists.append([m for m in cf if grid[m[0],m[1]] != -1])

    test_list = []
    test_lists = []
    for c in cf_lists:
        cf_val = []
        for p in c:
            try:
                test_list.append(grid[p[0], p[1]])
                cf_val.append(grid[p[0], p[1]])
            except:
                pass
        test_lists.append(set(cf_val))

    print sorted(test_list)

    # Check that all entrys are there
    print np.all(np.array(sorted(test_list)) == np.arange(grid_size))

    # Check that no two indices are in same list
    for i in range(len(test_lists)):
        for j in range(len(test_lists)):
            if i>j:
                assert(not bool(test_lists[i] & test_lists[j]))


def get_cf_3d(cube, nbsize, dim=3, pad=True):
    n_sets = nbsize**dim
    cf_lists = []

    cube_shape = np.shape(cube)
    cube_size = cube_shape[0] * cube_shape[1] * cube_shape[2]
    padded_dim = lcm([cube_shape[0], cube_shape[1], cube_shape[2], nbsize])
    print "padded_dim", padded_dim


    padded_cube = -1 * np.ones([padded_dim] * 3, dtype=int)
    padded_cube[:cube_shape[0], :cube_shape[1], :cube_shape[2]] = cube
    cube = padded_cube

    cf_lists = []
    for n in range(n_sets):
        cf = []
        p0 = np.array([n%nbsize, (n/nbsize)%2, n/(2*nbsize)])
        print p0
        cf = [p0 + np.array([nbsize * i, nbsize * j, nbsize * k]) for i in range(padded_dim/nbsize)\
                                                                  for j in range(padded_dim/nbsize)\
                                                                  for k in range(padded_dim/nbsize)]

        cf_lists.append([m for m in cf if cube[m[0],m[1],m[2]] != -1])


    test_list = []
    test_lists = []
    for cf_list in cf_lists:
        cf_val = []
        for idx in cf_list:
            test_list.append(cube[idx[0], idx[1], idx[2]])
            cf_val.append(cube[idx[0], idx[1], idx[2]])
        test_lists.append(set(cf_val))

    print sorted(test_list)

    # Check that all entrys are there
    print np.all(np.array(sorted(test_list)) == np.arange(cube_size))

    # Check that no two indices are in same list
    for i in range(len(test_lists)):
        for j in range(len(test_lists)):
            if i>j:
                assert(not bool(test_lists[i] & test_lists[j]))


if __name__ == "__main__":
    cube = np.arange(36*6).reshape(6,6,6)
    get_cf_3d(cube, 2)
   
    t0 = time.time()
    for i in range(2):
        grid = np.arange(10*7).reshape(10,7)
        nbsize = 8
        get_conflict_free_sets2d(grid, nbsize, pad=True, virtual=False)
    t1 = time.time()
    print t1 - t0
