import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import pdb
import copy

class GravGrid(object):
    def __init__(self):
        self.points = list()
        self.r_exp = 2.0
        self.g = 1.0
        
    def add_point(self, pos):
        assert(isinstance(pos, np.ndarray))
        assert(pos.shape[0], 3 )

        self.points.append(pos)
        return len(self.points) - 1

    def set_force(self, r_exp, g):
        """
        F_grav propto 1/r^(r_exp)
        """
        self.g = g
        self.r_exp = r_exp

    def get_dx2(self, j, dt):
        r_ij = np.array([self.points[i] for i in range(len(self.points)) if i != j]) -\
               np.array([self.points[j]] * (len(self.points) - 1))
        sum_ = 0
        for r in r_ij:
            sum_ += (r/np.linalg.norm(r))/np.dot(r, r)

        dx2 = -self.g * sum_ * dt**2
        return dx2

    def get_dx1(self, x0, x1, it, cycles):
        dx1 = (x1 - x0)/(1. + (0.03 * (1 - it/float(cycles))))
        return dx1

    def solve(self, cycles, dt):
        points0 = copy.deepcopy(self.points)

        plt.axis() 
        plt.ion()
        for it in range(cycles):
            
            for j in range(len(self.points)):
                if it == 0:
                    x_j1 = points0[j] + np.array([0., -1**j * 10., 0.0])

                x_j0 = points0[j]
                points0[j] = x_j1
                dx1 = self.get_dx1(x_j0, x_j1, it, cycles)
                dx2 = self.get_dx2(j, dt)/2. 

                #print j
                #print dx1, dx2, "\n"
                x_j1 = self.points[j] + dx1 + dx2/2. 
                self.points[j] = x_j1

            for p in self.points:
                plt.scatter(p[0], p[1])
            plt.pause(0.05)
            
            print self.points


if __name__ == "__main__":
    grav_grid = GravGrid()
    grav_grid.add_point(np.array([-100.0, 0., 0.])) 
    grav_grid.add_point(np.array([100.0, 0., 0.]))
    grav_grid.add_point(np.array([0.,0.,0.]))
    grav_grid.solve(1000, dt=0.00001)
