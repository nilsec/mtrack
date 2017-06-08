import os
import glob
import h5py
import numpy as np

def from_h5_to_h5stack(input_directory, output_file):
    input_files = glob.glob(input_directory + '*.h5')
    input_files.sort()
    
    for n, input_file in enumerate(input_files):
        f = h5py.File(input_file)
        data = f['exported_data'].value
        f.close()
            
        if n == 0:
            h5stack = np.zeros((data.shape[0], data.shape[1], len(input_files)), dtype=data.dtype)

        h5stack[:, :, n] = data[:, :, 0]
    
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
    input_directory = "/media/nilsec/d0/Data_MTs/Validation/raw/"
    ilastik_project = '/media/nilsec/d0/Data_MTs/ilastik/parallel_3.ilp'
    output_directory = 'media/nilsec/d0/gt_mt_data/probability_maps/validation/parallel'
    
    get_prob_map_ilastik(input_directory, output_directory, ilastik_source_directory, ilastik_project, verbose=True)
