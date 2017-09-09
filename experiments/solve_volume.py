import os
import sys
sys.path.append(os.path.join('..', '..'))

from gt_microtubules import solve
from timeit import default_timer as timer
#from gt_microtubules.preprocessing import extract_candidates
import json
import preprocessing
import cProfile, pstats, StringIO


def solve_volume(volume_dir,
                 start_edge_prior,
                 distance_factor,
                 orientation_factor,
                 comb_angle_factor,
                 selection_cost,
                 time_limit,
                 output_dir,
                 voxel_size):

    start = timer()

    print "Solve volume {}...".format(volume_dir)

    components = [f for f in os.listdir(volume_dir[:-1]) if f[-3:] == ".gt"]
    n_comp = len(components)

    print "with {} components...".format(n_comp)
    
    
    i = 0
    for cc in components:
        assert("phy" in cc) # assume physical coordinates

        cc_output_dir = os.path.join(output_dir[:-1], cc[:-3]) + "/"

        print "Solve cc {}/{}".format(i + 1, n_comp)
        solve.solve(os.path.join(volume_dir[:-1], cc),
                    start_edge_prior,
                    distance_factor,
                    orientation_factor,
                    comb_angle_factor,
                    selection_cost,
                    time_limit,
                    cc_output_dir,
                    voxel_size)
        i += 1
    
    end = timer()
    runtime = end - start

    stats = {"n_comps": n_comp,
             "runtime": runtime,
             "volume_dir": volume_dir}

    with open(output_dir + "stats.txt", "w+") as f:
        json.dump(stats, f)

def solve_bb_volume(bounding_box,
                    prob_map_stack,
                    gs,
                    ps,
                    distance_threshold,
                    start_edge_prior,
                    distance_factor,
                    orientation_factor,
                    comb_angle_factor,
                    selection_cost,
                    time_limit,
                    output_dir,
                    voxel_size):


    candidates = preprocessing.extract_candidates(prob_map_stack,
                                                  gs,
                                                  ps,
                                                  voxel_size,
                                                  bounding_box=bounding_box,
                                                  bs_output_dir=output_dir + "binary_stack/")

    g1 = preprocessing.candidates_to_g1(candidates, 
                                             voxel_size)

    g1_connected = preprocessing.connect_graph_locally(g1,
                                         distance_threshold)
    
    cc_list = g1_connected.get_components(min_vertices=4,
                                          output_folder = output_dir + "cc/")

    solve_volume(output_dir + "cc/",
                 start_edge_prior,
                 distance_factor,
                 orientation_factor,
                 comb_angle_factor,
                 selection_cost,
                 time_limit,
                 output_dir + "solution/",
                 voxel_size)


 
def test_volume_grid():
    prob_map_stack_file_perp_val = "/media/nilsec/d0/gt_mt_data/" +\
                               "probability_maps/validation/perpendicular/stack/stack.h5"

    prob_map_stack_file_par_val = "/media/nilsec/d0/gt_mt_data/" +\
                              "probability_maps/validation/parallel/stack/stack.h5" 

   
    prob_map_stack_file_perp_test = "/media/nilsec/d0/gt_mt_data/" +\
                               "probability_maps/test/perpendicular/stack/stack.h5"
    
    prob_map_stack_file_par_test = "/media/nilsec/d0/gt_mt_data/" +\
                               "probability_maps/test/parallel/stack/stack.h5"
 
  

    prob_map_stack = preprocessing.DirectionType(prob_map_stack_file_perp_test,
                                                 prob_map_stack_file_par_test)
  

    gs_0 = preprocessing.DirectionType(0.5, 0.5)
    ps_0 = preprocessing.DirectionType(0.6, 0.6)

    gs_1 = preprocessing.DirectionType(0.5, 0.5)
    ps_1 = preprocessing.DirectionType(0.4, 0.4)

    gs_2 = preprocessing.DirectionType(0.5, 0.5)
    ps_2 = preprocessing.DirectionType(0.4, 0.5)

    gs_3 = preprocessing.DirectionType(0.5, 0.5)
    ps_3 = preprocessing.DirectionType(0.5, 0.3)

    gs_4 = preprocessing.DirectionType(0.5, 0.5)
    ps_4 = preprocessing.DirectionType(0.4, 0.3)

    gs_5 = preprocessing.DirectionType(0.5, 0.5)
    ps_5 = preprocessing.DirectionType(0.3, 0.4)
 
 
    pp_list = [(gs_0, ps_0), (gs_1, ps_1), (gs_2, ps_2), (gs_3, ps_3), (gs_4, ps_4), (gs_5, ps_5)]  

    distance_threshold = 150

    start_edge_prior = 180.0
    distance_factor = 0.0
    orientation_factor = 15.0
    comb_angle_factor = 16.0
    selection_cost = -80.0
    output_dir = "/media/nilsec/d0/gt_mt_data/experiments/test_solve_{}/"
    time_limit = 1000
    voxel_size = [5.0, 5.0, 50.0]
    
    bounding_box = [300, 400]
  
    run = 0 
    for pp in pp_list:
         
        od = output_dir.format(run)

        run += 1
 
        if os.path.exists(od + "solution"):
            continue        
 
        solve_bb_volume(bounding_box,
                        prob_map_stack,
                        pp[0],
                        pp[1],
                        distance_threshold,
                        start_edge_prior,
                        distance_factor,
                        orientation_factor,
                        comb_angle_factor,
                        selection_cost,
                        time_limit,
                        od,
                        voxel_size)
        break

def get_performance():

    pr = cProfile.Profile()
    pr.enable()

    test_volume_grid()

    pr.disable()
    s = StringIO.StringIO()
    sortby = 'cumulative'
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.dump_stats("ps_stats_volume")

    


def solve_validation_volume():
    volume_dir = "/media/nilsec/d0/gt_mt_data/experiments/cc_dist89/"
    start_edge_prior = 180.0
    distance_factor = 0.0
    orientation_factor = 15.0
    comb_angle_factor = 16.0
    selection_cost = -80.0
    output_dir = "/media/nilsec/d0/gt_mt_data/experiments/validation_solve_dt89/"
    time_limit = 600
    voxel_size = [5.0, 5.0, 50.0]

    solve_volume(volume_dir,
                 start_edge_prior,
                 distance_factor,
                 orientation_factor,
                 comb_angle_factor,
                 selection_cost,
                 time_limit,
                 output_dir,
                 voxel_size)


if __name__ == "__main__":
    distance_threshold = 150
    start_edge_prior = 180.0
    distance_factor = 0.0
    orientation_factor = 15.0
    comb_angle_factor = 16.0
    selection_cost = -80.0
    output_dir = "/media/nilsec/d0/gt_mt_data/solve_volumes/volume_1/"
    time_limit = 1000
    voxel_size = [5.0, 5.0, 50.0]
    bounding_box = [300, 310]
    gs = preprocessing.DirectionType(0.5, 0.5)
    ps = preprocessing.DirectionType(0.4, 0.4)

    prob_map_stack_file_perp_test = "/media/nilsec/d0/gt_mt_data/" +\
                               "probability_maps/test/perpendicular/stack/stack.h5"
    
    prob_map_stack_file_par_test = "/media/nilsec/d0/gt_mt_data/" +\
                               "probability_maps/test/parallel/stack/stack.h5"
 
    prob_map_stack = preprocessing.DirectionType(prob_map_stack_file_perp_test,
                                                 prob_map_stack_file_par_test)
 
    solve_bb_volume(bounding_box,
                    prob_map_stack,
                    gs,
                    ps,
                    distance_threshold,
                    start_edge_prior,
                    distance_factor,
                    orientation_factor,
                    comb_angle_factor,
                    selection_cost,
                    time_limit,
                    output_dir,
                    voxel_size)
