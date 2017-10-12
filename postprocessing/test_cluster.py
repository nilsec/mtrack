import unittest
from cluster import line_segment_distance

class LineDistanceTestCase(unittest.TestCase):
    def runTest(self):
        line_p0 = (0,0,0)
        line_p1 = (10,0,0)

        point = (10,1,0)

        print line_segment_distance(line_p0, line_p1, point)

if __name__ == "__main__":
    unittest.main()
