class Chunk(object):
    def __init__(self, base_shape, shape):
        self.base_shape = base_shape
        self.shape = shape
        self.limits = [None, None, None]

    def get_limits(self, dim):
        return self.limits[dim]

    def set_limits(self, dim, min_lim, max_lim):
        self.limits[dim] = [min_lim, max_lim]


class Chunker(object):
    def __init__(self, volume_size, chunk_factor, overlap, voxel_size):
        self.chunks = []
        # voxel_size = [x,y,z]
        
    def reduce_chunk(self, chunk, voxel_size):
        max_dim = self.__get_max_dim(chunk.shape, voxel_size)
        
        reduced_shape = chunk.shape.copy()
        reduced_shape[max_dim] = int(np.ceil(chunk.shape[max_dim]/2.))

        limits_max_dim = chunk.get_limits(max_dim)

        reduced_limits = [limits_max_dim, limits_max_dim]
        reduced_limits[0][1] = reduced_limits[0][0] += reduced_shape[max_dim]
        reduced_limits[1][0] = reduced_limits[1][1] -= reduced_shape[max_dim] 
        
        reduced_chunk_0 = Chunk(chunk.base_shape,
                                reduced_shape)
        reduced_chunk_1 = Chunk(chunk.base_shape,
                                reduced_shape)

        reduced_chunk_0.set_limits(max_dim, reduced_limits[0])
        reduced_chunk_1.set_limits(max_dim, reduced_limits[1])
                 
        self.chunks.append()
            
    def __get_max_dim(self, bb_shape_vox, voxel_size):
        bb_shape_phys = bb_shape_vox * np.array(voxel_size[::-1])
        dimensions = sorted(enumerate(bb_shape_phys), key = lambda x: x[1])
        max_dim = dimensions[2][0]
        return max_dim


