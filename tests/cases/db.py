import unittest
import numpy as np
from numpy.random import randint

from mtrack.cores import DB, CoreBuilder
from mtrack.preprocessing import MtCandidate


class DBBaseTest(unittest.TestCase):
    def setUp(self):
        try:
            self.name_db = "unittest"
            self.collection = "run"
            self.db = DB()
            self.client = self.db.get_client(self.name_db,
                                             self.collection,
                                             overwrite=True)

            # Clean up all written data:
            self.addCleanup(self.db.get_client, self.name_db, self.collection, True)

        # Possible problem here is that no db instance is running:
        except Exception as e:
            print "Make sure that a DB instance is running before executing the test suite"
            raise e

        self.min_pos = 1
        self.max_pos = 101
        self.distance_threshold = 10
        self.voxel_size = np.array([5.,5.,50.])
 
        print "[Unittest]: Create mock candidates..."
        self.candidate_positions = []
        self.candidate_seperation = self.distance_threshold/2

        for x in np.arange(self.min_pos, self.max_pos, self.candidate_seperation):
            for y in np.arange(self.min_pos, self.max_pos, self.candidate_seperation):
                for z in np.arange(self.min_pos, self.max_pos, self.candidate_seperation):
                    self.candidate_positions.append(np.array([x,y,z]))

        
        self.n_candidates = len(self.candidate_positions)
        orientations = randint(self.min_pos, 
                               self.max_pos, 
                               size=(self.n_candidates, 3))
  

        self.candidate_orientations = []
        self.mock_candidates = []

        for i in range(self.n_candidates):
            identifier = i + 1 # To mirror id_0 = 1 in track.py

            if self.n_candidates - 10 > i > 10 and (i%3 == 0):
                partner_i = identifier + 1

            elif (self.mock_candidates and (self.mock_candidates[-1].partner_identifier == identifier)):
                partner_i = identifier - 1
                self.candidate_positions[i] = self.candidate_positions[i - 1]
            
            else:
                partner_i = -1
            
            pv = np.array(self.candidate_positions[i], dtype=float)
            
            ov = np.array(orientations[i], dtype=float)
            ov_normed = ov/np.linalg.norm(ov)
            self.candidate_orientations.append(ov_normed)

            candidate = MtCandidate(position=pv,
                                    orientation=ov_normed,
                                    identifier=identifier,
                                    partner_identifier=partner_i)

            self.mock_candidates.append(candidate)

        print "[Unittest]: Verify {} mock candidates...".format(self.n_candidates)
        for j in range(len(self.mock_candidates)):
            if self.mock_candidates[j].partner_identifier != -1:
                if self.mock_candidates[j + 1].partner_identifier ==\
                        self.mock_candidates[j].identifier:

                    self.assertEqual(self.mock_candidates[j].partner_identifier, 
                                     self.mock_candidates[j + 1].identifier)
                    self.assertTrue(np.all(self.mock_candidates[j].position ==\
                                    self.mock_candidates[j + 1].position))


class MonolithicDBTestCase(DBBaseTest):
    def runTest(self):
        """
        Populating the db is an expensive step and all other tests
        would need to repeat this. In order to avoid that
        this method tests all functionality of the 
        DB abstraction layer.
        """

        print "[Unittest]: Test write candidates..."        
        self.db.write_candidates(name_db=self.name_db,
                                 prob_map_stack_chunk=None,
                                 offset_chunk=None,
                                 gs=None,
                                 ps=None,
                                 voxel_size=self.voxel_size,
                                 id_offset=None,
                                 collection=self.collection,
                                 overwrite=True,
                                 candidates=self.mock_candidates)

        candidates_db = self.client.find({})
        
        ids_local = [candidate.identifier for candidate in self.mock_candidates]
        ids_db = []
        ids_db_partner = []
        for candidate in candidates_db:
            self.assertTrue(candidate["type"] == "vertex") # We have only vertices
            self.assertTrue(candidate["solved"] == False) # Check default inits before solving:
            self.assertTrue(candidate["selected"] == False)
            self.assertTrue(candidate["degree"] == 0) 
            self.assertTrue(candidate["id"] in ids_local)
            # Positions in db match with local positions times voxel size. Physical coords only in db:
            self.assertTrue(np.all(np.array([candidate["px"], candidate["py"], candidate["pz"]]) ==\
                                        self.mock_candidates[candidate["id"] - 1].position *\
                                            self.voxel_size))

            ids_db.append(candidate["id"])
            ids_db_partner.append(candidate["id_partner"])


        # All ids are present and we have no duplicates:
        self.assertTrue(len(set(ids_db)) == len(ids_db))
        self.assertTrue(sorted(ids_db) == range(1, self.n_candidates + 1))

        """
        Connect Candidates in chunks with overlap:
        """
        print "[Unittest]: Test connect candidates..."
        builder = CoreBuilder(volume_size=np.array([self.max_pos - self.min_pos] * 3) * self.voxel_size ,
                              core_size=np.array([10]*3) * self.voxel_size,
                              context_size=np.array([3*self.distance_threshold]*3) * self.voxel_size)

        cores = builder.generate_cores()

        for core in cores:
            self.db.connect_candidates(self.name_db,
                                       self.collection,
                                       x_lim=core.x_lim_context,
                                       y_lim=core.y_lim_context,
                                       z_lim=core.z_lim_context,
                                       distance_threshold=self.distance_threshold * self.voxel_size[0])

        candidates_connected_db = self.client.find({})
        
        edges_db = []
        for candidate in candidates_connected_db:
            if candidate["type"] == "edge":
                self.assertFalse(candidate["selected"])
                self.assertFalse(candidate["solved"])
                self.assertEqual(candidate["time_selected"], [])
                self.assertEqual(candidate["by_selected"], [])

                id0 = candidate["id0"]
                id1 = candidate["id1"]

                edges_db.append((id0, id1))

                self.assertTrue(id0 in ids_db)
                self.assertTrue(id1 in ids_db)
                
                vertices_in_edge = self.client.find({"id": {"$in": [id0, id1]}})
                vertices_in_edge = [v for v in vertices_in_edge]

                self.assertTrue(len(vertices_in_edge) == 2)

                self.assertTrue(vertices_in_edge[0]["id_partner"] != vertices_in_edge[1]["id"])
                self.assertTrue(vertices_in_edge[1]["id_partner"] != vertices_in_edge[0]["id"])

                for v in vertices_in_edge:
                    self.assertEqual(v["degree"], 0)
                    self.assertFalse(v["solved"])
                    self.assertFalse(v["selected"])
                    self.assertEqual(v["time_selected"], [])
                    self.assertEqual(v["by_selected"], [])
         
        self.assertTrue(len(edges_db)>0)
        self.assertEqual(len(edges_db), len(set(edges_db)))

        print "[Unittest]: Test get_g1..."
        min_pos = np.array([self.min_pos]*3) * self.voxel_size
        max_pos = np.array([self.max_pos]*3) * self.voxel_size

        x_lim_full = {"min": min_pos[0],
                      "max": max_pos[0]}

        y_lim_full = {"min": min_pos[1],
                      "max": max_pos[1]}

        z_lim_full = {"min": min_pos[2],
                      "max": max_pos[2]}

        g1, index_map = self.db.get_g1(name_db=self.name_db,
                                       collection=self.collection,
                                       x_lim=x_lim_full,
                                       y_lim=y_lim_full,
                                       z_lim=z_lim_full,
                                       query_edges=True)

        
        self.assertEqual(g1.get_number_of_vertices(), self.n_candidates)
        self.assertEqual(len(edges_db), g1.get_number_of_edges())

        for v in g1.get_vertex_iterator():
            self.assertFalse(g1.get_vertex_property("selected", v))
            self.assertFalse(g1.get_vertex_property("solved", v))

            self.assertTrue(tuple(g1.get_position(v)) in\
                           [tuple(p * self.voxel_size) for p in self.candidate_positions])

            self.assertTrue(tuple(g1.get_orientation(v)) in\
                           [tuple(o * self.voxel_size) for o in self.candidate_orientations])

            db_vertices = self.client.find({"id": index_map[v]})
            self.assertTrue(db_v.count() == 1)
            for v_db in db_vertices:
                self.assertTrue( np.all(g1.get_position(v) ==\
                                 np.array([v_db["px"], v_db["py"], v_db["pz"]]) ) )

                self.assertTrue( np.all(g1.get_orientation(v) ==\
                                 np.array([v_db["ox"], v_db["oy"], v_db["oz"]])))

                self.assertEqual(v_db["id_partner"], g1.get_partner(v))

        for e in g1.get_edge_iterator():
            self.assertFalse(g1.get_edge_property("selected", e.source(), e.target()))
            self.assertFalse(g1.get_edge_property("solved", e.source(), e.target()))

            id_v0 = e.source()
            id_v1 = e.target()

            id_db_v0 = index_map[int(id_v0)]
            id_db_v1 = index_map[int(id_v1)]

            db_edges = self.client.find(
                            {"$and": [
                                      {"id0": {"$in": [id_db_v0, id_db_v1]}},  
                                      {"id1": {"$in": [id_db_v0, id_db_v1]}}
                                     ]}
                            )

            self.assertTrue(db_edges.count() == 1)

            for e_db in db_edges:
                self.assertFalse(e_db["selected"])
                self.assertFalse(e_db["solved"])
                self.assertEqual(e_db["time_selected"], [])
                self.assertEqual(e_db["by_selected"], [])
                self.assertEqual(e_db["type"], "edge")

            db_vertices0 = self.client.find({"id": id_db_v0})
            db_vertices1 = self.client.find({"id": id_db_v1})
            self.assertTrue(db_vertices0.count() == 1)
            self.assertTrue(db_vertices1.count() == 1)

            for v_db in db_vertices0:
                self.assertTrue( np.all(g1.get_position(id_v0) ==\
                                 np.array([v_db["px"], v_db["py"], v_db["pz"]]) ) )

                self.assertTrue( np.all(g1.get_orientation(id_v0) ==\
                                 np.array([v_db["ox"], v_db["oy"], v_db["oz"]])))

                self.assertEqual(v_db["id_partner"], g1.get_partner(id_v0))
                self.assertEqual(v_db["type"], "vertex")

            for v_db in db_vertices1:
                self.assertTrue( np.all(g1.get_position(id_v1) ==\
                                 np.array([v_db["px"], v_db["py"], v_db["pz"]]) ) )

                self.assertTrue( np.all(g1.get_orientation(id_v1) ==\
                                 np.array([v_db["ox"], v_db["oy"], v_db["oz"]])))

                self.assertEqual(v_db["id_partner"], g1.get_partner(id_v1))
                self.assertEqual(v_db["type"], "vertex")
        
if __name__ == "__main__":
    unittest.main()
