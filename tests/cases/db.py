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
        self.max_pos = 100
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
        self.candidate_orientations = randint(self.min_pos, 
                                              self.max_pos, 
                                              size=(self.n_candidates, 3))
 
  

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


            candidate = MtCandidate(position=np.array(self.candidate_positions[i], dtype=float),
                                    orientation=np.array(self.candidate_orientations[i], dtype=float),
                                    identifier=identifier,
                                    partner_identifier=partner_i)

            self.mock_candidates.append(candidate)

        print "[Unittest]: Verify {} mock candidates...".format(self.n_candidates)
        for j in range(len(self.mock_candidates)):
            if self.mock_candidates[j].partner_identifier != -1:
                if self.mock_candidates[j + 1].partner_identifier == self.mock_candidates[j].identifier:
                    self.assertEqual(self.mock_candidates[j].partner_identifier, self.mock_candidates[j + 1].identifier)
                    self.assertTrue(np.all(self.mock_candidates[j].position == self.mock_candidates[j + 1].position))


class WriteConnectCandidatesTestCase(DBBaseTest):
    def runTest(self):
        """
        Write Candidates:
        """

        self.db.write_candidates(name_db=self.name_db,
                                 prob_map_stack_chunk=None,
                                 offset_chunk=None,
                                 gs=None,
                                 ps=None,
                                 voxel_size=[5.,5.,50.],
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
                                   self.mock_candidates[candidate["id"] - 1].position * self.voxel_size))
            ids_db.append(candidate["id"])
            ids_db_partner.append(candidate["id_partner"])


        # All ids are present and we have no duplicates:
        self.assertTrue(len(set(ids_db)) == len(ids_db))
        self.assertTrue(sorted(ids_db) == range(1, self.n_candidates + 1))

        """
        Connect Candidates in chunks with overlap:
        """

        builder = CoreBuilder(volume_size=np.array([self.max_pos] * 3) * self.voxel_size ,
                              core_size=np.array([10]*3) * self.voxel_size,
                              context_size=np.array([3*self.distance_threshold]*3) * self.voxel_size)

        cores = builder.generate_cores()

        print "[Unittest]: Connect candidates..." 
        for core in cores:
            self.db.connect_candidates(self.name_db,
                                       self.collection,
                                       x_lim=core.x_lim_context,
                                       y_lim=core.y_lim_context,
                                       z_lim=core.z_lim_context,
                                       distance_threshold=self.distance_threshold * self.voxel_size[0])

        candidates_connected_db = self.client.find({})
        
        for candidate in candidates_connected_db:
            if candidate["type"] == "edge":
                id0 = candidate["id0"]
                id1 = candidate["id1"]
                
                vertices_in_edge = self.client.find({"id": {"$in": [id0, id1]}})
                vertices_in_edge = [v for v in vertices_in_edge]
                self.assertTrue(len(vertices_in_edge) == 2)
                self.assertTrue(vertices_in_edge[0]["id_partner"] != vertices_in_edge[1]["id"])
                self.assertTrue(vertices_in_edge[1]["id_partner"] != vertices_in_edge[0]["id"])

            
if __name__ == "__main__":
    unittest.main()
