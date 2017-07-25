import pyted
import numpy as np
import pickle
import os
import process_solution
import json

def evaluate_lines(tracing_line_dir, 
                   rec_line_dir, 
                   distance_threshold, 
                   voxel_size,
                   dimensions,
                   correction):

    print "Get line list..."
    tracing_lines = process_solution.get_line_list(tracing_line_dir)
    rec_lines = process_solution.get_line_list(rec_line_dir)

    print "Get tracing volume..."
    tracing_volume = process_solution.get_volume(tracing_lines, dimensions, correction)
    print "Get rec volume..."
    rec_volume = process_solution.get_volume(rec_lines, dimensions, correction)

    report = evaluate_skeleton(tracing_volume,
                               rec_volume,
                               distance_threshold,
                               voxel_size,
                               from_skeleton=False)

    return report
    

def evaluate_skeleton(tracing_volume, reconstruction_volume, distance_threshold, 
                      voxel_size, background_label=0.0, report_ted=True, 
                      report_rand=True, report_voi=True, verbose=True, 
                      from_skeleton=True):

    print "Start Evaluation..."
        
    parameters = pyted.Parameters()
    parameters.report_ted = report_ted
    parameters.report_rand = report_rand
    parameters.report_voi = report_voi
    parameters.distance_threshold = distance_threshold
    parameters.from_skeleton = from_skeleton
    parameters.gt_background_label = background_label
    parameters.rec_background_label = background_label
    parameters.have_background = True

    print "Initialize TED..."
    ted = pyted.Ted(parameters)
    print "Create Report...\n"
    
    report = ted.create_report(tracing_volume.astype(np.uint32), 
                               reconstruction_volume.astype(np.uint32),
                               np.array([voxel_size[2], voxel_size[1],voxel_size[0]]).astype(np.float64)) # Jan changed order to z, y, x in new ted version.
    
    if verbose:
        print("\nTED report:")
        for (k,v) in report.iteritems():
            print("\t" + k.ljust(20) + ": " + str(v))

    return report

if __name__ == "__main__":
    trace_dir = "/media/nilsec/d0/gt_mt_data/test_tracing/lines_v17_cropped"
    rec_dir = "/media/nilsec/d0/gt_mt_data/experiments/selection_cost_grid0404_solve_4/lines"
    distance_threshold = 100
    voxel_size = [5.,5.,50.]
    dimensions=[1025, 1025, 101]
    correction = np.array([0,0,300])

    report = evaluate_lines(trace_dir, 
                            rec_dir, 
                            distance_threshold,
                            voxel_size,
                            dimensions,
                            correction)

    json.dump(report, open("./report.json", "w+"))
