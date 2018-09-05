import unittest
import numpy as np
from mtrack.graphs import cost_converter, graph_converter, g1_graph, g2_solver
from mtrack import g2_to_g1_solution
from mtrack.solve import solve
from mtrack.preprocessing import MtCandidate
from mtrack.cores import DB, Core
from mtrack.track import solve_core
import pdb

class LineSetUp(unittest.TestCase):
    def setUp(self):
        """
        Here we set up a problem instance where the solution should
        be a straight line spanning multiple cores.
        """
    
        self.line_points = [np.array([0.,0.,10. + j]) for j in range(100)]
        self.line_orientations = [np.array([0.,0.,1.])] * len(self.line_points)
        self.line_start = self.line_points[0]
        self.line_end = self.line_points[-1]

        self.wrong_candidate_positions = [np.array([0., 3., 10. + j]) for j in np.random.randint(0,100,30)]
        self.wrong_candidate_orientations = [np.array([1.0, 0.0, 0.0])] * len(self.wrong_candidate_positions)

        self.candidates = []
        for i in range(len(self.line_points)):
            candidate = MtCandidate(self.line_points[i],
                                    self.line_orientations[i],
                                    identifier=i+1,
                                    partner_identifier=-1)

            self.candidates.append(candidate)

        i0 = len(self.line_points) + 1
        for j in range(len(self.wrong_candidate_positions)):
            candidate = MtCandidate(self.wrong_candidate_positions[j],
                                    self.wrong_candidate_orientations[j],
                                    identifier=i0 + j,
                                    partner_identifier=-1) 
        
        try:
            self.name_db = "unittest"
            self.collection = "Line"
            self.db = DB()
            self.client = self.db.get_client(self.name_db,
                                             self.collection,
                                             overwrite=True)

            self.addCleanup(self.db.get_client, self.name_db, self.collection, True)
        except Exception as e:
            print "Make sure that a DB instance is running befoer executing the test suite"
            raise e

        self.db.write_candidates(name_db=self.name_db,
                                 prob_map_stack_chunk=None,
                                 offset_chunk=None,
                                 gs=None,
                                 ps=None,
                                 voxel_size=[1.,1.,1.],
                                 id_offset=0,
                                 collection=self.collection,
                                 overwrite=True,
                                 candidates=self.candidates)

        self.roi_x = {"min": 0, "max": 20}
        self.roi_y = {"min": 0, "max": 20}
        self.roi_z = {"min": 0, "max": 120}

        self.db.connect_candidates(name_db=self.name_db, 
                                collection=self.collection,
                                x_lim=self.roi_x,
                                y_lim=self.roi_y,
                                z_lim=self.roi_z,
                                distance_threshold=3.5)

class SolveTestCase(LineSetUp):
    def runTest(self):
        cores = []
       
        line_range = self.line_end - self.line_start
        
        i = 0
        for z in range(int(self.line_start[2]), int(self.line_end[2]) + 10, 10):
            x_lim = self.roi_x
            y_lim = self.roi_y
            z_lim = {"min": z - 5, "max": z + 5}

            core = Core(x_lim=x_lim,
                        y_lim=y_lim,
                        z_lim=z_lim,
                        context=[8,8,8],
                        core_id=i,
                        nbs=[i+1, i-1]
                        )
            cores.append(core)
            i += 1

        graph = self.db.get_client(self.name_db,
                                   self.collection)

        i = 0
        for core in cores:
            print "Solve Core {}".format(i)
            solve_core(core,
                       self.name_db,
                       self.collection,
                       cc_min_vertices=0,
                       start_edge_prior=10,
                       selection_cost=-10000,
                       distance_factor=0.0,
                       orientation_factor=100.0,
                       comb_angle_factor=100.0,
                       time_limit=None,
                       voxel_size=[1.,1.,1.],
                       backend="Gurobi")
            i += 1

            #graph.update_many({"solved": True}, {"$set": {"solved": False}})

        g1, index_map = self.db.get_g1(self.name_db,
                                       self.collection,
                                       self.roi_x,
                                       self.roi_y,
                                       self.roi_z)

        selected, index_map = self.db.get_selected(self.name_db,
                                        self.collection,
                                        x_lim=self.roi_x,
                                        y_lim=self.roi_y,
                                        z_lim=self.roi_z)

        for v in selected.get_vertex_iterator():
            self.assertTrue(len(selected.get_incident_edges(v)) <= 2)
               

        ccs_selected = selected.get_components(min_vertices=0,
                                               output_folder=None,
                                               return_graphs=True)

        g1_line = g1_graph.G1(len(self.line_points))
        i = 0
        for v in g1_line.get_vertex_iterator():
            g1_line.set_position(v, self.line_points[i])
            g1_line.set_orientation(v, self.line_orientations[i])
            g1_line.set_partner(v, -1)
            if i != 99:
                g1_line.add_edge(i, i+1)
            i += 1
        
        self.assertTrue(selected.get_number_of_vertices() == 100)
        self.assertTrue(selected.get_number_of_edges() == 99)
        self.assertTrue(len(ccs_selected) == 1) 
        self.assertTrue(g1_line == selected)

if __name__ == "__main__":
    unittest.main()
