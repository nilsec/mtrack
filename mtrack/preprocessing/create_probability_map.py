import os
import glob
import h5py
import numpy as np
import pdb

def ilastik_get_prob_map(raw,
                         output_dir,
                         ilastik_source_dir,
                         ilastik_project,
                         file_extension,
                         h5_dset=None,
                         label=0):

    h5_extensions = [".h5", ".hdf5", ".hdf"]
    other_extensions = [".png", ".tiff", ".tif"]

    allowed_extensions = h5_extensions + other_extensions
    assert(file_extension in allowed_extensions)

    if os.path.isdir(raw):
        stack_path = ilastik_prob_map_from_zslices(raw,
                                                   output_dir,
                                                   ilastik_source_dir,
                                                   ilastik_project,
                                                   file_extension,
                                                   label)

    elif os.path.isfile(raw):
        assert(h5_dset is not None)

        is_h5 = np.any([raw.endswith(ext) for ext in h5_extensions])
        assert(is_h5)

        stack_path = ilastik_prob_map_from_h5stack(raw,
                                                   h5_dset,
                                                   output_dir,
                                                   ilastik_source_dir,
                                                   ilastik_project,
                                                   label)

    return stack_path


def stack_to_chunks(input_stack, output_dir, chunks):
    """
    Chunk probability map h5 stack according to a list
    of chunk objects. See mtrack/preprocessing/chunker.py
    The input stack is expected to have 3 dimensions ordered 
    according to z,y,x convention.
    """

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    f0 = h5py.File(input_stack)
    data = f0["exported_data"]
    assert(len(data.shape) == 3)
 
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
 

def ilastik_prob_map_from_zslices(input_dir,
                                  output_dir, 
                                  ilastik_source_dir, 
                                  ilastik_project, 
                                  file_extension='.png',
                                  label=0):
    """
    The input dir should contain z-wise slices of the raw data
    with the file extension specified. This can be for example
    png or tiff. The resulting h5 prob map stack will be 
    3 dimensional in (z,y,x) convention consistent with the
    output of ilastik_prob_map_from_h5stack.
    """
    
    output_stack_dir = output_dir + '/stack'
    if not os.path.exists(output_stack_dir):
        os.makedirs(output_stack_dir)

    input_files = glob.glob(input_dir + '/*' + file_extension)
    cmd = ilastik_source_dir + "/run_ilastik.sh --headless --project=" + ilastik_project + " "
    
    for input_file in input_files:
        cmd += input_file + " "

    pdb.set_trace()
    cmd += " --output_filename_format=" + output_dir + "/{nickname}.h5"
    os.system(cmd)
    
    #Write h5 stack:
    output_stack = output_stack_dir + "/stack.h5"
    from_h5_to_h5stack(output_dir, output_stack, label)

    return output_stack


def ilastik_prob_map_from_h5stack(h5stack,
                                  h5_dset,
                                  output_dir,
                                  ilastik_source_dir,
                                  ilastik_project,
                                  label=0):

    """
    This function generates a probability map h5 stack
    from a 3D input volume given as an h5 stack.
    The raw data dimensions are expected to follow a dimensional
    ordering of z,y,x
    
    Ilastik adds a 4th dimension to a 3D dataset indicating the labels. 
    This function grabs only the label indicated and 
    returns a 3D array consisting of prob(x=label).
    """
   
    output_dir += "/stack"

    f_in = h5py.File(h5stack)
    dset = f_in[h5_dset]
    input_shape = dset.shape
    f_in.close()

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cmd = ilastik_source_dir + "/run_ilastik.sh --headless --project=" + ilastik_project + " "
    cmd += h5stack
    cmd += h5_dset
    cmd += " --output_filename_format=" + output_dir + "/{nickname}.hdf5"

    os.system(cmd)
    
    output_stack = os.path.join(output_dir, os.path.basename(h5stack))
    f_out = h5py.File(output_stack, "r")
    # exported_data is the dset naming convention by ilastik
    dset = f_out["exported_data"]
    output_shape = dset.shape

    assert(np.all(np.array(input_shape) == np.array(output_shape)[:-1]))
    output_stack_corrected = output_stack.replace(".hdf5","_corrected.hdf5") 
    f_out_corrected = h5py.File(output_stack_corrected, "w")
    f_out_corrected.create_dataset("exported_data", data=dset[:,:,:,label])
    f_out.close()
    f_out_corrected.close()

    # Clean up
    os.remove(output_stack)

    return output_stack_corrected


def from_h5_to_h5stack(input_dir, output_file, label=0):
    """
    Take a collection of h5 z-slices and stack them
    along the z dimensions. The resulting dataset
    is ordered according to z,y,x convention.
    """
    input_files = glob.glob(input_dir + '/*.h5')
    input_files.sort()

    for n, input_file in enumerate(input_files):
        f = h5py.File(input_file)
        data = f['exported_data'].value
        f.close()
            
        if n == 0:
            h5stack = np.zeros((len(input_files), data.shape[1], data.shape[0]), dtype=data.dtype)

        h5stack[n, :, :] = data[:, :, label]
    
    f = h5py.File(output_file, 'w')
    f.create_dataset('exported_data', data=h5stack)
    f.close()


if __name__ == "__main__":
    base_path = "/media/nilsec/d0/gt_mt_data/data_cremi/MTTest_CremiTraining_Aligned"
    output_dir = "/media/nilsec/d0/gt_mt_data/probability_maps/cremi/sampleA_aligned"
    h5_dset = "/volumes/raw"
    """
    ilastik_get_prob_map(base_path + "/sample_A.augmented.0.hdf5",
                         output_dir,
                         ilastik_source_dir="/usr/local/src/ilastik-1.2.0rc10-Linux",
                         ilastik_project="/media/nilsec/d0/gt_mt_data/data_cremi/ilastik/perp.ilp",
                         file_extension=".hdf5",
                         h5_dset=h5_dset)
    """

    raw = "/media/nilsec/d0/gt_mt_data/data/Validation/raw_0_250"
    output_dir = "/media/nilsec/d0/gt_mt_data/data/probability_maps/l3_val_0_250_test"
    ilastik_get_prob_map(raw,
                         output_dir,
                         ilastik_source_dir="/usr/local/src/ilastik-1.2.0rc10-Linux",
                         ilastik_project="/media/nilsec/d0/gt_mt_data/ilastik/perpendicular.ilp",
                         file_extension=".png",
                         h5_dset=None) 
