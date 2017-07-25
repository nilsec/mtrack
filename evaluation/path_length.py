mport skeleton_tools
import numpy as np
from scipy.ndimage import binary_dilation
import os


def enlarge_binary_map(binary_map, marker_size_voxel=1, voxel_size=None, marker_size_physical=None):
    """
    Enlarge existing regions in a binary map.
    Parameters
    ----------
        binary_map: numpy array
            matrix with zeros, in which regions to be enlarged are indicated with a 1 (regions can already
            represent larger areas)
        marker_size_voxel: int
            enlarged region have a marker_size (measured in voxels) margin added to
            the already existing region (taking into account the provided voxel_size). For instance a marker_size_voxel
            of 1 and a voxel_size of [2, 1, 1] (z, y, x) would add a voxel margin of 1 in x,y-direction and no margin
            in z-direction.
        voxel_size:     tuple, list or numpy array
            indicates the physical voxel size of the binary_map.
        marker_size_physical: int
            if set, overwrites the marker_size_voxel parameter. Provides the margin size in physical units. For
            instance, a voxel_size of [20, 10, 10] and marker_size_physical of 10 would add a voxel margin of 1 in
            x,y-direction and no margin in z-direction.
    Returns
    ---------
        binary_map: matrix with 0s and 1s of same dimension as input binary_map with enlarged regions (indicated with 1)
    """
    if voxel_size is None:
        voxel_size = (1, 1, 1)
    voxel_size = np.asarray(voxel_size)
    if marker_size_physical is None:
        voxel_size /= np.min(voxel_size)
        marker_size = marker_size_voxel
    else:
        marker_size = marker_size_physical
    binary_map = np.logical_not(binary_map)
    binary_map = distance_transform_edt(binary_map, sampling=voxel_size)
    binary_map = binary_map <= marker_size
    binary_map = binary_map.astype(np.uint8)
    return binary_map

def anisotropic_dilatation(bin_mask, zdil=1, xydil=10):
    print bin_mask.shape
    zdilarray = binary_dilation(bin_mask, iterations=zdil)
    xydil_list = []
    new_array = np.empty_like(bin_mask)
    for z_index in range(bin_mask.shape[2]):
        zsection = bin_mask[:, :, z_index].squeeze()
        # print zsection.shape, 'zsectionshape'
        xydilarray = binary_dilation(zsection, iterations=xydil)
        # xydilarray = np.expand_dims(xydilarray, 2)
        # xydil_list.append(xydilarray)
        # print xydilarray.shape
        new_array[:, :, z_index] = xydilarray
    # xydilwhole = np.hstack(xydil_list)
    new_dil = np.maximum(new_array, zdilarray)
    return new_dil


def get_false_positive_skeletons(gt_skeleton, predict_skeleton, dilatation_iter=2, anisotropic=False):
    bb_low, bb_upper = gt_skeleton.get_bounding_box()
    bin_mask = gt_skeleton.from_skeletons_to_binary_mask(bb_upper)
    if anisotropic:
        bin_mask = anisotropic_dilatation(bin_mask)
    else:
        bin_mask = binary_dilation(bin_mask, iterations=dilatation_iter)
    bin_mask = bin_mask <= 0
    for skeleton in predict_skeleton.skeleton_list:
        skeleton.crop_with_binmask(bin_mask)


def get_false_negative_skeletons(gt_skeleton, predict_skeleton, dilatation_iter=2, anisotropic=False):
    bb_low, bb_upper = gt_skeleton.get_bounding_box()
    bin_mask = predict_skeleton.from_skeletons_to_binary_mask(bb_upper)
    if anisotropic:
        bin_mask = anisotropic_dilatation(bin_mask)
    else:
        bin_mask = binary_dilation(bin_mask, iterations=dilatation_iter)
    bin_mask = bin_mask <= 0
    for skeleton in gt_skeleton.skeleton_list:
        skeleton.crop_with_binmask(bin_mask)


L1GRAPH = False
ANISOTROPIC = True

# skeletonfilename = '/raid/julia/projects/microtubules/may2017recollection/ILPresults/newskeleton/' \
#                    'ILP_results/drosophila_validation_final/knossos/fitted_models_chunk0.nml'
if L1GRAPH:
    skeletonfilename = '/raid/julia/documents/Dropbox/Doktorarbeit/2017/projects/' \
                       'NIPSmicrotubules/skeletons/nilspredictions.nml'
else:
    skeletonfilename = '/raid/julia/documents/Dropbox/Doktorarbeit/2017/projects/' \
                       'NIPSmicrotubules/skeletons/stateoftheartprediction_cleaned.nml'

gt_filename = '/raid/julia/documents/Dropbox/Doktorarbeit/2017/projects/' \
              'NIPSmicrotubules/skeletons/testsingleneuron_v1.nml'



# gt_filename = '/raid/julia/projects/microtubules/may2017recollection/ILPresults/' \
#               'newskeleton/ILP_results/drosophila_validation_final/knossos/GT.nml'

if L1GRAPH:
    outputdirectory = '/raid/julia/projects/microtubules/may2017recollection/metricpathlength/l1prediction/'
else:
    outputdirectory = '/raid/julia/projects/microtubules/may2017recollection/metricpathlength/stateoftheartprediction/'

if not os.path.exists(outputdirectory):
    os.makedirs(outputdirectory)

dilatation_iter = 5 # evaluation metric parameter


sk_congt = skeleton_tools.SkeletonContainer()
sk_congt.read_from_knossos_nml(gt_filename, voxel_size=[5, 5, 50])
gt_pathlength = sk_congt.calculate_total_phys_length()

sk_conpred = skeleton_tools.SkeletonContainer()
sk_conpred.read_from_knossos_nml(skeletonfilename, voxel_size=[5, 5, 50])
get_false_positive_skeletons(sk_congt, sk_conpred, dilatation_iter=dilatation_iter, anisotropic=ANISOTROPIC)

sk_conpred.write_to_knossos_nml(outputdirectory + 'falsepositives.nml')
falsepositive_pathlength = sk_conpred.calculate_total_phys_length()

# Currently the skeletons are modified in place, that 's why we have to reload the original
# skeleton (because of missing copy operator).

sk_conpred = skeleton_tools.SkeletonContainer()
sk_conpred.read_from_knossos_nml(skeletonfilename, voxel_size=[5, 5, 50])

get_false_negative_skeletons(sk_congt, sk_conpred, dilatation_iter=dilatation_iter, anisotropic=ANISOTROPIC)
sk_congt.write_to_knossos_nml(outputdirectory + 'falsenegatives.nml')

falsenegative_pathlength = sk_congt.calculate_total_phys_length()

print 'path length of ground truth skeleton', gt_pathlength/1000.
print 'paht length of false positives', falsepositive_pathlength/1000.
print 'path length of false negative', falsenegative_pathlength/1000.
