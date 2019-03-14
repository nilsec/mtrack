import unittest
import numpy as np
from mtrack.graphs import g1_graph
try:
    import matplotlib.pyplot as plt
except:
    pass

class SmallCircleGraphTestCase(unittest.TestCase):
    def setUp(self):
        self.g1_vertices_N = 10
        self.g1 = g1_graph.G1(self.g1_vertices_N)

class SmallSquareGraphTestCase(unittest.TestCase):
        # orientation = (1, 0, 0)
        # --->

        #  1----2
        #  |    |
        #  |    |
        #  0----3
 
    def setUp(self):
        self.g1_vertices_N = 4
        
        self.g1 = g1_graph.G1(self.g1_vertices_N)
        
        self.g1.set_position(0, np.array([0.0, 0.0, 0.0]))
        self.g1.set_position(1, np.array([0.0, 1.0, 0.0]))
        self.g1.set_position(2, np.array([1.0, 1.0, 0.0]))
        self.g1.set_position(3, np.array([1.0, 0.0, 0.0]))

        self.g1.add_edge(0, 1)
        self.g1.add_edge(1, 2)
        self.g1.add_edge(2, 3)
        self.g1.add_edge(3, 0)

        self.orientation = np.array([1.0, 0.0, 0.0])

        for v in self.g1.get_vertex_iterator():
            self.g1.set_orientation(v, self.orientation)

class IncreasingEnergyGraphTestCase(unittest.TestCase):
        # orientation = (1, 0, 0)
        # --->

        #  1----2
        #  |    
        #  |    
        #  0
 
        # vs:
        #
        # 1 
        # | \
        # |  \
        # 0   2

    def setUp(self):
        self.g1_vertices_N = 3
        
        self.g1 = g1_graph.G1(self.g1_vertices_N)
        self.g1.set_position(0, np.array([0.0, 0.0, 0.0]))
        self.g1.set_position(1, np.array([0.0, 1.0, 0.0]))
        self.g1.set_position(2, np.array([1.0, 1.0, 0.0]))
        self.g1.add_edge(0, 1)
        self.g1.add_edge(1, 2)

        self.orientation = np.array([1.0, 0.0, 0.0])

        for v in self.g1.get_vertex_iterator():
            self.g1.set_orientation(v, self.orientation)

        self.g1_high = g1_graph.G1(self.g1_vertices_N)
        self.g1_high.set_position(0, np.array([0.0, 0.0, 0.0]))
        self.g1_high.set_position(1, np.array([0.0, 1.0, 0.0]))
        self.g1_high.set_position(2, np.array([1.0, 0.0, 0.0]))
        self.g1_high.add_edge(0, 1)
        self.g1_high.add_edge(1, 2)


def generate_bending(angle, radius):
    g1_vertices_N = 3
    g1 = g1_graph.G1(g1_vertices_N)
    g1.set_position(0, np.array([0.0, 0.0, 0.0]))
    g1.set_position(1, np.array([0.0, radius, 0.0]))

    x = np.sin(angle) * radius
    y = np.cos(angle) * radius
    g1.set_position(2, np.array([0.0, y, x]))
    g1.add_edge(0, 1)
    g1.add_edge(0, 2)

    orientation = np.array([1.0, 0.0, 0.0])

    for v in g1.get_vertex_iterator():
        g1.set_orientation(v, orientation)

    return g1, x, y

class ClockTestCase(unittest.TestCase):
    def runTest(self):
        radius = 1
        comb_angle_factor = 1.0
        angles = []
        costs = []
        x_1 = []
        y_1 = []
        for angle in np.arange(0, 2*np.pi, 0.01):
            g1, x, y = generate_bending(angle, radius)
            edge_combination_cost = g1.get_edge_combination_cost(comb_angle_factor)
            x_1.append(x)
            y_1.append(y)
            e01 = g1.get_edge(0,1)
            e12 = g1.get_edge(0,2)
            angles.append(angle)
            costs.append(edge_combination_cost[(e01, e12)])

        radius = 10
        comb_angle_factor = 1.0
        angles_10 = []
        costs_10 = []
        x_10 = []
        y_10 = []
        for angle in np.arange(0, 2*np.pi, 0.01):
            g1, x, y = generate_bending(angle, radius)
            edge_combination_cost = g1.get_edge_combination_cost(comb_angle_factor)
            x_10.append(x)
            y_10.append(y)
            e01 = g1.get_edge(0,1)
            e12 = g1.get_edge(0,2)
            angles_10.append(angle)
            costs_10.append(edge_combination_cost[(e01, e12)])



        f, (ax1, ax2) = plt.subplots(1, 2)
        ax1.semilogy(angles, costs, label="r=1")
        ax1.semilogy(angles_10, costs_10, label="r=10")
        ax1.set_xlabel("angle")
        ax1.set_ylabel("cost")
        #ax1.axvline(np.pi/2)
        #ax1.axvline(np.pi)
        ax1.legend()
        ax1.grid()
        ax2.scatter(x_1, y_1)
        ax2.scatter(x_10, y_10)
        ax2.set_xlabel("x")
        ax2.set_ylabel("y")
        ax2.text(-10, 0, str(np.arcsin(-10/10))[:3])
        ax2.text(0, -10, str(np.arcsin(0/10))[:3])
        ax2.text(10, 0, str(np.arcsin(10/10))[:3])
        ax2.text(0, 10, str(np.arcsin(0/10))[:3])
        plt.show()


class CompareEdgeCombinationCostTestCase(IncreasingEnergyGraphTestCase):
    def runTest(self):
        comb_angle_factor = 10.0

        edge_combination_cost_low = self.g1.get_edge_combination_cost(comb_angle_factor)
        edge_combination_cost_high = self.g1_high.get_edge_combination_cost(comb_angle_factor)

        e_01 = self.g1.get_edge(0,1)
        e_12 = self.g1.get_edge(1,2)
        start_edge = self.g1.START_EDGE

        e_01_high = self.g1_high.get_edge(0,1)
        e_12_high = self.g1_high.get_edge(1,2)
        start_edge_high = self.g1_high.START_EDGE

        print edge_combination_cost_low, edge_combination_cost_high
        self.assertTrue(edge_combination_cost_low[(e_01, e_12)] < edge_combination_cost_high[(e_01_high,e_12_high)])


class GetEdgeCombinationCostTestCase(SmallSquareGraphTestCase):
    def runTest(self):
        comb_angle_factor =  10.0

        edge_combination_cost = self.g1.get_edge_combination_cost(comb_angle_factor)

        e_01 = self.g1.get_edge(0,1)
        e_12 = self.g1.get_edge(1,2)
        e_23 = self.g1.get_edge(2,3)
        e_30 = self.g1.get_edge(3,0)
        start_edge = self.g1.START_EDGE

        # Possible Combinations:
        #   1. Each edge with start edge: 4
        #   2.      e_01, e_12
        #           e_12, e_23
        #           e_23, e_30
        #           e_30, e_01
        #
        #   --> 8 edge pairs
        self.assertEqual(len(edge_combination_cost), 8)

        self.assertEqual(edge_combination_cost[(e_01, e_12)], edge_combination_cost[(e_12, e_23)])
        self.assertEqual(edge_combination_cost[(e_12, e_23)], edge_combination_cost[(e_23, e_30)])
        self.assertEqual(edge_combination_cost[(e_23, e_30)], edge_combination_cost[(e_01, e_30)])
 
        self.assertEqual(edge_combination_cost[(e_01, start_edge)], 0.0)
        self.assertEqual(edge_combination_cost[(e_12, start_edge)], 0.0)
        self.assertEqual(edge_combination_cost[(e_23, start_edge)], 0.0)
        self.assertEqual(edge_combination_cost[(e_30, start_edge)], 0.0)


if __name__ == "__main__":
    unittest.main()
