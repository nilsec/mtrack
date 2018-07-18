from mtrack.cores import DB, CoreBuilder
from mtrack.preprocessing import g1_to_nml
import os
import numpy as np

def retrieve(name_db,
             collection,
             x_lim,
             y_lim,
             z_lim,
             voxel_size,
             output_path):

    db = DB()
    g1_selected, index_map = db.get_selected(name_db,
                                             collection,
                                             x_lim=x_lim,
                                             y_lim=y_lim,
                                             z_lim=z_lim)

    print "Validate solution..."
    for v in g1_selected.get_vertex_iterator():
        assert(len(g1_selected.get_incident_edges(v)) <= 2), "Retrieved graph has branchings"
    print "...No violations"

    g1_to_nml(g1_selected, 
              output_path,
              knossos=True,
              voxel_size=voxel_size)

if __name__ == "__main__":
    
    roi = [[300,1100], [300,1100], [2,52]]
    roi_volume_size = np.array([r[1] - r[0] for r in roi]) * np.array([5.,5.,50.])
    builder = CoreBuilder(volume_size=roi_volume_size,
                          core_size=np.array([150,150,15])*np.array([5.,5.,50.]),
                          context_size=np.array([100,100,10])*np.array([5.,5.,50.]),
                          offset=np.array([r[0] for r in roi]) * np.array([5.,5.,50.]))
    """    
    cores = builder.generate_cores()
    for core in cores:
        print "Retrieve core {}".format(core.id)
        retrieve("cremi_validation_a0_1",
                 "microtubules",
                {"min":core.x_lim_core["min"], "max":core.x_lim_core["max"]},
                {"min":core.y_lim_core["min"], "max":core.y_lim_core["max"]},
                {"min":core.z_lim_core["min"], "max":core.z_lim_core["max"]},
                [5.,5.,50.],
                "/media/nilsec/d0/gt_mt_data/mtrack/grid_A+/grid_1/core_{}.nml".format(core.id))

    for core in cores:
        print "Retrieve context {}".format(core.id)
        retrieve("cremi_validation_a0_1",
                 "microtubules",
                {"min":core.x_lim_context["min"], "max":core.x_lim_context["max"]},
                {"min":core.y_lim_context["min"], "max":core.y_lim_context["max"]},
                {"min":core.z_lim_context["min"], "max":core.z_lim_context["max"]},
                [5.,5.,50.],
                "/media/nilsec/d0/gt_mt_data/mtrack/grid_A+/grid_1/context_{}.nml".format(core.id))
    """
    print "Retrieve full roi" 
    retrieve("cremi_validation_a0_1_rerun",
                 "microtubules",
                {"min":roi[0][0]*5, "max":roi[0][1]*5},
                {"min":roi[1][0]*5, "max":roi[1][1]*5},
                {"min":roi[2][0]*50, "max":roi[2][1]*50},
                [5.,5.,50.],
                "/media/nilsec/d0/gt_mt_data/mtrack/grid_A+/grid_1/roi_0.nml")

