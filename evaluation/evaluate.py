import pyted
import process_tracings
import numpy as np
import pickle
import os


def evaluate_skeleton(tracing_volume, reconstruction_volume, distance_threshold, 
                      voxel_size, background_label=0.0, report_ted=True, 
                      report_rand=True, report_voi=True, verbose=True, 
                      from_skeleton=False):

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
    
    rec_ted = np.zeros(np.shape(reconstruction_volume))
    report = ted.create_report(tracing_volume.astype(np.uint32), 
                               reconstruction_volume.astype(np.uint32),
                               np.array([voxel_size[2], voxel_size[1],voxel_size[0]]).astype(np.float64)) # Jan changed order to z, y, x in new ted version.
    
    if verbose:
        print("\nTED report:")
        for (k,v) in report.iteritems():
            print("\t" + k.ljust(20) + ": " + str(v))

    return report
