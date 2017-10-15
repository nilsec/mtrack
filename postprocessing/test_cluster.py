import unittest
from cluster import line_segment_distance
from cluster import Cluster

class LineDistanceTestCase(unittest.TestCase):
    def runTest(self):
        line_p0 = (0,0,0)
        line_p1 = (10,0,0)

        point = (10,1,0)

        print "ls distance: ", line_segment_distance(line_p0, line_p1, point)


class OverlapTestCase(unittest.TestCase):
    def runTest(self):
        bb_min_0 = [6,0,0]
        bb_max_0 = [10,10,10]

        bb_min_1 = [4,8,6]
        bb_max_1 = [20,20,20]        

        print bool(Cluster.do_overlap(bb_min_0, bb_max_0, bb_min_1, bb_max_1))

if __name__ == "__main__":
    unittest.main()
