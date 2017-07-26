import skeleton_tools
import numpy as np
from scipy.ndimage import binary_dilation
from scipy.ndimage.morphology import distance_transform_edt
import os
import postprocessing
import json
import matplotlib.pyplot as plt


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


def get_false_positive_skeletons(gt_skeleton, predict_skeleton, tolerance, voxel_size):
    bb_low, bb_upper = gt_skeleton.get_bounding_box()
    bin_mask = gt_skeleton.from_skeletons_to_binary_mask(bb_upper)
    bin_mask = enlarge_binary_map(bin_mask, marker_size_physical=tolerance, voxel_size=voxel_size)
    bin_mask = bin_mask <= 0
    for skeleton in predict_skeleton.skeleton_list:
        skeleton.crop_with_binmask(bin_mask)


def get_false_negative_skeletons(gt_skeleton, predict_skeleton, tolerance, voxel_size):
    bb_low, bb_upper = gt_skeleton.get_bounding_box()
    bin_mask = predict_skeleton.from_skeletons_to_binary_mask(bb_upper)
    bin_mask = enlarge_binary_map(bin_mask, marker_size_physical=tolerance, voxel_size=voxel_size)
    bin_mask = bin_mask <= 0
    for skeleton in gt_skeleton.skeleton_list:
        skeleton.crop_with_binmask(bin_mask)

def evaluate(gt_filename, solution_dirs, tolerance, voxel_size, base_output_dir):
    result_dirs = []
    
    for sol in solution_dirs:
        base_path = base_output_dir + sol.split("/")[-2]
        print "Process ", sol, "..."
        result_dirs.append(base_path)

        combined_solution = base_path + "/combined.nml"
        postprocessing.combine_knossos_solutions(sol, combined_solution)
        
        sk_congt = skeleton_tools.SkeletonContainer()
        sk_congt.read_from_knossos_nml(gt_filename, voxel_size=voxel_size)
        gt_pathlength = sk_congt.calculate_total_phys_length()

        sk_conpred = skeleton_tools.SkeletonContainer()
        sk_conpred.read_from_knossos_nml(combined_solution, voxel_size=voxel_size)
        get_false_positive_skeletons(sk_congt, 
                                     sk_conpred, 
                                     tolerance=tolerance, 
                                     voxel_size=voxel_size)

        sk_conpred.write_to_knossos_nml(base_path + '/false_positives.nml')
        falsepositive_pathlength = sk_conpred.calculate_total_phys_length()

        sk_conpred = skeleton_tools.SkeletonContainer()
        sk_conpred.read_from_knossos_nml(combined_solution, voxel_size=voxel_size)

        get_false_negative_skeletons(sk_congt, 
                                     sk_conpred, 
                                     tolerance=tolerance, 
                                     voxel_size=voxel_size)

        sk_congt.write_to_knossos_nml(base_path + '/false_negatives.nml')

        falsenegative_pathlength = sk_congt.calculate_total_phys_length()

        results = {"gt_pl": gt_pathlength/1000.,
                   "fp_pl": falsepositive_pathlength/1000.,
                   "fn_pl": falsenegative_pathlength/1000.}

        json.dump(results, open(base_path + "/results.json", "w+"))

        print 'path length of ground truth skeleton', gt_pathlength/1000.
        print 'path length of false positives', falsepositive_pathlength/1000.
        print 'path length of false negative', falsenegative_pathlength/1000. 

def get_precision_recall(json_dump):
    res = json.load(open(json_dump, "r"))

    fp = res["fp_pl"]
    fn = res["fn_pl"]
    tot = res["gt_pl"]
    tp = tot - fn

    precision = tp/(tp + fp)
    recall = tp/(tp + fn)
    return (precision, recall)

def plot_precision_recall(pr_list, label=None, title=None):
    plt.grid()

    precision = [pr[0] for pr in pr_list]
    recall = [pr[1] for pr in pr_list]

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    #plt.xlim(0, 1.0)
    #plt.ylim(0, 1.0)

    plt.plot(recall, precision, linestyle="", marker="x", label=label)
    plt.legend()
    if title is not None:
        plt.title(title)

def process_evaluation(base_eval_dir_list, dir_tag, label_list=[]):
    n = 0
    for base_eval_dir in base_eval_dir_list:
        eval_dirs = [f for f in os.listdir(base_eval_dir) if dir_tag in f]
        print eval_dirs

        pr_list = []
        for d in eval_dirs:
            base_file_dir = os.path.join(base_eval_dir, d)
            files = os.listdir(os.path.join(base_eval_dir, d))
            print files
            for f in files:
                if f.endswith(".json"):
                    pr = get_precision_recall(os.path.join(base_file_dir,f))
                    pr_list.append(pr)

        print pr_list

        if label_list:
            plot_precision_recall(pr_list, label=label_list[n], title=dir_tag)
        else:
            plot_precision_recall(pr_list, title=dir_tag)
        n += 1

    plt.show()
    
    
def nips_paper_eval():
    for tolerance in [100, 50, 10]:

        skeletonfilename = "../postprocessing/sol_sc_5.nml"

        gt_filename = '/media/nilsec/d0/gt_mt_data/test_tracing/v17_cropped.nml'
        base_output_dir = "/media/nilsec/d0/gt_mt_data/experiments/path_length_eval_%s/" % tolerance

        bp = "/media/nilsec/d0/gt_mt_data/experiments/"
        solution_dirs_selection_cost = [bp +\
            "selection_cost_grid0404_solve_%s/solution" % j for j in range(10)]

        solution_dirs_start_prior = [bp +\
            "start_prior_grid0404_solve_%s/solution" % j for j in range(10)]

        solution_dirs = solution_dirs_selection_cost + solution_dirs_start_prior

        evaluate(gt_filename, solution_dirs, tolerance, [5,5,50], base_output_dir)


if __name__ == "__main__":
    #nips_paper_eval()
    bp = "/media/nilsec/d0/gt_mt_data/experiments/path_length_eval_{}"
    process_evaluation([bp.format(100), bp.format(50), bp.format(10)], "start_prior", ["tolerance 100", "tolerance 50", "tolerance 10"])
