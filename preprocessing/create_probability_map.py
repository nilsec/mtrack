import os
import glob
import h5py
import numpy as np

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
    input_files = glob.glob(input_directory + '*.h5')
    input_files.sort()
    
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
    

def get_prob_map_ilastik(input_directory, output_directory, ilastik_source_directory, ilastik_project, verbose=False):

    assert(os.path.exists(input_directory))
    
    if verbose:
        print "\nCreate Ilastik probability map...\n" 
        print "Input Data: " + input_directory + "\n"
        print "Ilastik Project: " + ilastik_project + "\n"
        print "Ilastik Source: " + ilastik_source_directory + "\n\n\n"
            

    output_path = output_directory + 'stack/'

    try:
        os.makedirs(os.path.dirname(output_path))
    except:
        pass

    input_files = glob.glob(input_directory + '*.png')
    
    #Get Probability map:
    cmd = ilastik_source_directory + "/run_ilastik.sh --headless --project=" + ilastik_project + " "
    
    for input_file in input_files:
        cmd += input_file + " "

    cmd += " --output_filename_format=" + output_directory + "{nickname}.h5"
    os.system(cmd)
    
    #Write h5 stack:
    output_stack = output_path + "stack.h5"
    from_h5_to_h5stack(output_directory, output_stack)

    
    if verbose:
        print "Ilastik probability map written to " + output_stack


if __name__ == "__main__":
    ilastik_source_directory = "/usr/local/src/ilastik-1.2.0rc10-Linux/"
    input_directory = "/media/nilsec/d0/Data_MTs/Test/raw_split/"
    #ilastik_project = '/media/nilsec/d0/Data_MTs/ilastik/parallel_3.ilp'
    #ilastik_project = '/media/nilsec/d0/gt_mt_data/ilastik/perpendicular.ilp'
    ilastik_project = "/media/nilsec/d0/Data_MTs/ilastik/perp.ilp"
    output_directory = '/media/nilsec/d0/gt_mt_data/probability_maps/test/perpendicular/'
    
    #get_prob_map_ilastik(input_directory, output_directory, ilastik_source_directory, ilastik_project, verbose=True)
    from_h5_to_h5stack("/media/nilsec/m1/gt_mt_data/probability_maps/validation/parallel/",
                       "/media/nilsec/m1/gt_mt_data/probability_maps/validation/parallel/stack/stack_corrected.h5")
