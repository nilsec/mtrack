import numpy as np
import copy
import pdb


class Chunk(object):
    def __init__(self, voxel_size):
        """
        limits = [x_lim, y_lim, z_lim]
        voxel_size = [x,y,z]
        shape = [z,y,x]
        """
        self.limits = [None, None, None] # Convention: this is saved always in vox coordinates
        self.shape = np.array([None, None, None])
        self.voxel_size = np.array(voxel_size)
        self.id = None
        
    def get_shape(self, phys=False):
        if phys:
            return self.shape * self.voxel_size[::-1]
        else:
            return self.shape

    def get_limits(self, dim, phys=False):
        """
        Get limits along one dimension. If phys 
        is true return in physical coordinates.
        """
        if phys:
            return self.limits[dim] * self.voxel_size[dim]
        else:
            return self.limits[dim]

    def set_limits(self, dim, min_lim, max_lim, phys=False):
        """
        Set limits along one dimension. If phys =True
        it is assumed that the limits are given as physical
        coordinates and divided by voxel size before saving.
        """
        if phys:
            min_lim = int(np.ceil(min_lim/float(self.voxel_size[dim])))
            max_lim = int(np.ceil(max_lim/float(self.voxel_size[dim])))

        self.limits[dim] = np.array([min_lim, max_lim])
        self.shape[2 - dim] = max_lim - min_lim
        

class Chunker(object):
    def __init__(self, volume_shape, max_chunk_shape, voxel_size, overlap, offset=np.array([0,0,0]), phys=False):
        """
        volume_shape: shape of base volume you want to chunk [z,y,x]
        max_chunk_size: maximum chunk shape after chunking [z,y,x]
        offset: offset of base volume [x,y,z]
        overlap: overlap in [x,y,z]

        If phys is set to true it is assumed that all values
        are given in physical space and will be divided by voxel size for
        processing.
        """
        
        scale = np.array([1.,1.,1.])
        if phys:
            scale = np.array(voxel_size)
            
        self.max_chunk_shape = max_chunk_shape/scale[::-1]
        self.volume_shape = volume_shape/scale[::-1]
        self.overlap = overlap/scale

        self.voxel_size = np.array(voxel_size)
        self.volume = self.__init_volume(volume_shape, offset)
        self.chunks = []
        self.n_chunks = 0

    def __init_volume(self, volume_shape, offset):
        volume = Chunk(self.voxel_size)
        
        for dim in range(3):
            volume.set_limits(dim,
                              offset[dim], 
                              volume_shape[2-dim] + offset[dim], 
                              phys=False)

        return volume

    def chunk(self, axis=None):
        self.__chunk(self.volume, axis=axis)
        return self.chunks    
 
    def __chunk(self, chunk, axis=None):
        """
        Recursively half the size of the chunk along dimension
        that differs maximally from specified
        max chunk size until each chunk is smaller or equal
        to that size. You can force a chunk axis by specifying axis.
        """
        chunk_shape_phys = chunk.get_shape(phys=True)
        overlap_phys = self.overlap * self.voxel_size

        if axis is None:
            max_dim = sorted(enumerate(chunk_shape_phys - self.max_chunk_shape * self.voxel_size), key = lambda x: x[1])[2][0]
        else:
            max_dim = axis

        max_dim_shape_phys = chunk_shape_phys[max_dim]
        limits_max_dim_phys = chunk.get_limits(2 - max_dim, phys=True)

        reduced_dim_shape_phys = int(np.ceil(max_dim_shape_phys/2.)) + int(overlap_phys[2 - max_dim]/2.)
        
        reduced_limits_max_dim_phys = [[limits_max_dim_phys[0], limits_max_dim_phys[0] + reduced_dim_shape_phys],
                                       [limits_max_dim_phys[1] - reduced_dim_shape_phys, limits_max_dim_phys[1]]]

        reduced_chunk_0 = Chunk(self.voxel_size)
        for dim in range(3):
            reduced_chunk_0.set_limits(dim, 
                                       chunk.limits[dim][0],
                                       chunk.limits[dim][1])

        reduced_chunk_0.set_limits(2 - max_dim, 
                                   reduced_limits_max_dim_phys[0][0],
                                   reduced_limits_max_dim_phys[0][1],
                                   phys=True)


        reduced_chunk_1 = Chunk(self.voxel_size)
        for dim in range(3):
            reduced_chunk_1.set_limits(dim, 
                                       chunk.limits[dim][0],
                                       chunk.limits[dim][1])
 
        reduced_chunk_1.set_limits(2-max_dim,
                                   reduced_limits_max_dim_phys[1][0],
                                   reduced_limits_max_dim_phys[1][1],
                                   phys=True)
        

        if np.all(reduced_chunk_0.shape <= self.max_chunk_shape) and np.all(reduced_chunk_1.shape <= self.max_chunk_shape):

            reduced_chunk_0.id = self.n_chunks
            self.n_chunks += 1
            reduced_chunk_1.id = self.n_chunks
            self.n_chunks += 1

            self.chunks.append(reduced_chunk_0)
            self.chunks.append(reduced_chunk_1)
        else:
            self.__chunk(reduced_chunk_0, axis=axis)
            self.__chunk(reduced_chunk_1, axis=axis)

