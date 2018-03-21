import os
import glob
import h5py
import numpy as np


def stack_to_chunks(input_stack, output_dir, chunks):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)


    f0 = h5py.File(input_stack)
    data = f0["exported_data"]
 
    for chunk in chunks:
        limits = chunk.limits
        chunk_data = np.array(data[limits[2][0]:limits[2][1], 
                                   limits[1][0]:limits[1][1], 
                                   limits[0][0]:limits[0][1]])
 
        f = h5py.File(output_dir + "/chunk_{}.h5".format(chunk.id), 'w')
        f.create_dataset("exported_data", data=chunk_data)
        f["exported_data"].attrs.create("chunk_id", chunk.id)
        f["exported_data"].attrs.create("limits", limits)
        f.close()
    f0.close()
 
def slices_to_chunks(input_dir, output_dir, chunks):
    """
    The input dir should contain the probability
    maps of the volume z-slice wise in h5 format.
    This function produces h5 files of chunk sizes
    corresponding to the given chunk list.
    """

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    z_slices = glob.glob(input_dir + '/*.h5')
    z_slices.sort()

    f0 = h5py.File(z_slices[0])
    s0 = f0["exported_data"].value
    f0.close()
 
    for chunk in chunks:
        limits = chunk.limits
        slices = z_slices[limits[2][0]:limits[2][1]]
 
        h5_chunk = np.zeros(chunk.shape, dtype=s0.dtype)

        z = 0
        for z_slice in slices:
            f = h5py.File(z_slice)
            data = f["exported_data"].value
            f.close()
            
            h5_chunk[z,:,:] = data[limits[0][0]:limits[0][1], limits[1][0]:limits[1][1], 0]
            z += 1
        
        f = h5py.File(output_dir + "/chunk_{}.h5".format(chunk.id), 'w')
        f.create_dataset("exported_data", data=h5_chunk)
        f["exported_data"].attrs.create("chunk_id", chunk.id)
        f["exported_data"].attrs.create("limits", limits)
        f.close()

def from_h5_to_h5stack(input_directory, output_file):
    input_files = glob.glob(input_directory + '/*.h5')
    input_files.sort()

    if len(input_files) == 1:
        os.rename(input_files[0], output_file)
    
    else:
        for n, input_file in enumerate(input_files):
            f = h5py.File(input_file)
            data = f['exported_data'].value
            f.close()
                
            if n == 0:
                h5stack = np.zeros((len(input_files), data.shape[1], data.shape[0]), dtype=data.dtype)

            h5stack[n, :, :] = data[:, :, 0]
        
        f = h5py.File(output_file, 'w')
        f.create_dataset('exported_data', data=h5stack)
        f.close()
        

def get_prob_map_ilastik(output_directory, 
                         ilastik_source_directory, 
                         ilastik_project, 
                         input_directory=None,
                         file_extension='.hdf5',
                         h5_input_path=None,
                         verbose=False):

    if h5_input_path == "None":
        """
        Config reader workaround
        """
        h5_input_path = None

    assert(not (h5_input_path is None) or not (input_directory is None))
    
    if verbose:
        print "\nCreate Ilastik probability map...\n" 
        print "Input Data: " + input_directory + "\n"
        print "Ilastik Project: " + ilastik_project + "\n"
        print "Ilastik Source: " + ilastik_source_directory + "\n\n\n"
            

    output_path = output_directory + '/stack'
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    """
    if h5_input_path is None:
        input_files = glob.glob(input_directory + '*' + file_extension)
    else:
        input_files = [h5_input_path]
    #Get Probability map:
    cmd = ilastik_source_directory + "/run_ilastik.sh --headless --project=" + ilastik_project + " "
    
    for input_file in input_files:
        cmd += input_file + " "

    cmd += " --output_filename_format=" + output_directory + "/{nickname}.h5"
    os.system(cmd)
    """
    #Write h5 stack:
    print output_directory
    output_stack = output_path + "/stack.h5"
    from_h5_to_h5stack(output_directory, output_stack)

    if verbose:
        print "Ilastik probability map written to " + output_stack
    
    return output_stack
