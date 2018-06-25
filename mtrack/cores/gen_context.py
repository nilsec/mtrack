import numpy as np
from fractions import gcd 
import time


def lcm(denominators):
    return reduce(lambda a,b: a*b // gcd(a,b), denominators)


def get_core_cfs(core_size, context_size, volume_size, pad=True, debug=True):

    nbsizes = np.ceil(2*context_size.astype(float)/core_size) + 1
    effective_volume_size = volume_size - 2 * context_size
    cube_size = effective_volume_size/core_size
    cube = np.arange(reduce(lambda x,y: x*y, cube_size)).reshape(cube_size)

    return get_cfs(cube, nbsizes.astype(int), pad, debug)



def get_cfs(cube, nbsizes, pad=True, debug=True):
    """
    Generates conflict free sets for a 3d cube with arbitrary
    shape and neighbourhood size. Large inbalance of dimensions and/or 
    large neighbourhood sizes lead to more lists with lesser entries each.
    Not each list is guaranteed to have the same amount of entries. This is 
    only true if each axis has the same dimension and this dimension is divisible
    by the neighborhood size.
    """

    dim = 3
    nbsize = max(nbsizes)
    n_lists = nbsize**dim
    cf_lists = []

    cube_shape = np.shape(cube)
    cube_size = cube_shape[0] * cube_shape[1] * cube_shape[2]
    padded_dim = lcm([cube_shape[0], cube_shape[1], cube_shape[2], nbsize])

    # Pad cube to have side lengths of lcm(cube_shape, neighborhood)
    padded_cube = -1 * np.ones([padded_dim] * 3, dtype=int)
    padded_cube[:cube_shape[0], :cube_shape[1], :cube_shape[2]] = cube
    cube = padded_cube

    cf_lists = []
    for n in range(n_lists):
        cf = []
        # Iterate over all offsets:
        p0 = np.array([n%nbsize, (n/nbsize)%nbsize, n/(nbsize**2)%nbsize])
        cf = [p0 + np.array([nbsize * i, nbsize * j, nbsize * k]) for i in range(padded_dim/nbsize)\
                                                                  for j in range(padded_dim/nbsize)\
                                                                  for k in range(padded_dim/nbsize)]

        tmp = [cube[m[0], m[1], m[2]] for m in cf if cube[m[0],m[1],m[2]] != -1]
        if tmp:
            cf_lists.append(tmp)

    if debug:
        check_conflicts(cf_lists, cube_size)

    return cf_lists


def check_conflicts(cf_lists, max_id):
    all_entries = np.array(sorted([i for cf in cf_lists for i in cf]))
    expected_entries = np.arange(max_id)
    assert(np.all(all_entries == expected_entries))

    n_lists = len(cf_lists)
    for i in range(n_lists):
        for j in range(n_lists):
            if i>j:
                assert(not (set(cf_lists[i]) & set(cf_lists[j])))
