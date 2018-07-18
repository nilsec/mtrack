import unittest
import numpy as np
from numpy.random import randint

from mtrack.cores import DB
from mtrack.preprocessing import MtCandidate


class DBBaseTest(unittest.TestCase):
    def setUp(self):
        try:
            self.name_db = "unittest"
            self.collection = "run"
            self.db = DB()
            self.client = db.get_client(self.name_db,
                                        self.collection,
                                        overwrite=True)

            # Clean up all written data:
            unittest.addCleanup(db.get_client, self.name_db, self.collection, True)

        # Possible problem here is that no db instance is running:
        except Exception as e:
            raise e

        self.n_candidates = 1000
        self.min_pos = 1
        self.max_pos = 100

        self.candidate_positions = randint(self.min_pos, 
                                           self.max_pos, 
                                           size=(self.n_candidates, 3))

        self.candidate_orientations = randint(self.min_pos, 
                                              self.max_pos, 
                                              size=(self.n_candidates, 3))

        self.mock_candidates = []

        for i in range(self.n_candidates):
            identifier = i + 1 # To mirror id_0 = 1 in track.py

            if i > 10 and (i%3 == 0):
                partner_i = identifier + 1

            elif self.mock_candiates[-1].partner_identifier == identifier:
                partner_i = identifier - 1
            
            else:
                partner_i = -1


            candidate = MtCandidate(position=np.array(self.candidate_position[i], dtype=float),
                                    orientation=np.array(self.candidate_orientation[i], dtype=float),
                                    identifier=identifier,
                                    partner_identifier=partner_i)

            self.mock_candidates.append(candidate)

        self.distance_threshold = self.max_pos/4.


if __name__ == "__main__":
    unittest.main()
