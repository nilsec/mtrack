from evaluation import evaluate
import os

def evaluate_grid(grid_path, 
                  tracing_file, 
                  chunk_size,
                  distance_tolerance,
                  dummy_cost,
                  edge_selection_cost,
                  pair_cost_factor,
                  max_edges,
                  voxel_size=[5.,5.,50.],
                  time_limit=600):

    grids = [os.path.join(grid_path, f) for f in os.listdir(grid_path) if os.path.isdir(os.path.join(grid_path, f))]

    for grid in grids:
        solution = os.path.join(grid, "solution/volume.gt")
        if os.path.exists(os.path.join(grid, "solution/evaluation_0")):
            print "Skip grid " + os.path.basename(grid)
            continue
        print "Evaluate grid " + os.path.basename(grid)
        evaluate(tracing_file,
                 solution,
                 chunk_size,
                 distance_tolerance,
                 dummy_cost,
                 edge_selection_cost,
                 pair_cost_factor,
                 max_edges,
                 voxel_size,
                 time_limit=time_limit)

if __name__ == "__main__":
    tracing_file = "/media/nilsec/d0/gt_mt_data/DL3-tracings/validation/master_300_329.nml"
    chunk_size=10
    distance_tolerance=100.0
    dummy_cost=1000000
    edge_selection_cost=-10.0
    pair_cost_factor=1.0
    max_edges=3

    evaluate_grid("/media/nilsec/d0/gt_mt_data/solve_volumes/grid_2",
                  tracing_file=tracing_file,
                  chunk_size=chunk_size,
                  distance_tolerance=distance_tolerance,
                  dummy_cost=dummy_cost,
                  edge_selection_cost=edge_selection_cost,
                  pair_cost_factor=pair_cost_factor,
                  max_edges=max_edges)
