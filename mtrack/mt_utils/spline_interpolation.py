from scipy.interpolate import CubicSpline
import numpy as np
try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
except:
    pass


def get_energy_from_ordered_points(ordered_points, n_samples=1000):
    cs = get_spline(ordered_points)
    d1,d2,d3 = get_derivatives(cs, up_to_order=3)
    energy,_,_ = get_energy(cs,d1,d2,d3, np.linspace(0, len(ordered_points) - 1, n_samples))
    return energy


def get_spline(ordered_points):
    """
    ordered_points: list of arrays of points of arbitrary dimension. They need to be ordered
                    as the generated spline respects the ordering of the points.

    returns: The cubic spline interpolation as a scipy.PPoly instance. This can
             be translated in actual interpolation points via 
             x = cs(xx).T[0], y = cs(xx).T[1], z = cs(xx).T[2], ... with 
             xx = np.linspace(t0, t1, N)
    """
    t = range(len(ordered_points))
    y = ordered_points

    cs = CubicSpline(t,y)
    return cs


def get_derivatives(cubic_spline, up_to_order=3):
    """
    Calculates the derivatives of a cubic spline up to 
    the provided order.
    """
    return [cubic_spline.derivative(k) for k in range(1, up_to_order + 1)]


def get_energy(cs_d0, cs_d1, cs_d2, cs_d3, t, plot=False):
    """
    See Frenet-Serret Formulas on wiki

    Calculates the energy of a given cubic spline by
    integrating torsion and curvature along
    the line.

    cs_d0: cubic_spline
    cs_d1: 1st derivative
    cs_d2: 2nd derivative
    cs_d3: 3rd derivative
    t: np.array - parametrization of the curve, 
       determines the accuracy of the 
       result.

    returns: The energy of the curve (scalar), 
             the curvature as a function of t
             the torsion as a function of t
    """
    epsilon = 10**(-8)

    d0_t = np.array(cs_d0(t))
    d1_t = np.array(cs_d1(t))
    d2_t = np.array(cs_d2(t))
    d3_t = np.array(cs_d3(t))

    curvature_t = np.linalg.norm(np.cross(d1_t,d2_t), axis=1)/np.clip(np.linalg.norm(d1_t, axis=1)**3, epsilon, None)
    torsion_t = np.linalg.det(np.dstack([d1_t, d2_t, d3_t]))/np.clip(np.linalg.norm(np.cross(d1_t, d2_t),axis=1)**2, epsilon, None)

    energy = np.sum(curvature_t**2 + torsion_t**2) * np.abs(t[0] - t[1])
    if plot:
        plot_spline(d0_t)
    return energy, curvature_t, torsion_t


def plot_spline(d0_t):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    x = d0_t.T[0]
    y = d0_t.T[1]
    z = d0_t.T[2]
    ax.plot(x, y, z, label='spline')
    ax.legend()
    plt.show()


def test_helix(n=100):
    """
    Helix has constant curvature and torsion.
    """

    theta_max = 10 * np.pi
    theta = np.linspace(0, theta_max, n)
    y = [np.array([x, np.sin(x), np.cos(x)]) for x in theta]

    cs = get_spline(y)
    d1,d2,d3 = get_derivatives(cs)
    energy, curvature, torsion = get_energy(cs, d1, d2, d3, np.linspace(0,10*np.pi,200))
    print("Energy", energy)
    print("Curvature", curvature)
    print("Torsion", torsion)
    
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    x = cs(theta).T[0]
    y = cs(theta).T[1]
    z = cs(theta).T[2]
    ax.plot(x, y, z, label='helix')
    ax.legend()
    plt.show()

def test_line(n=100):
    """
    A line has 0 curvature and torsion.
    """

    t = np.linspace(0, 10, n)
    y = [np.array([5,x,5]) for x in t]

    cs = get_spline(y)
    d1,d2,d3 = get_derivatives(cs)
    energy, curvature, torsion = get_energy(cs, d1, d2, d3, np.linspace(0,10,500))
    print("Energy", energy)
    print("Curvature", curvature)
    print("Torsion", torsion)
    
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    x = cs(t).T[0]
    y = cs(t).T[1]
    z = cs(t).T[2]
    ax.plot(x, y, z, label='line')
    ax.legend()
    plt.show()

if __name__ == "__main__":
    test_line()
