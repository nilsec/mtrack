import numpy as np
import h5py
import skimage.measure
import random
from scipy import ndimage
import matplotlib.pyplot as plt
import pickle

diam_out = 24 # Outer diameter of microtubule in nm.


class MtCandidate:
    def __init__(self, position, orientation, identifier, partner_identifier=-1):
        self.position = position
        self.orientation = orientation
        self.identifier = identifier
        self.partner_identifier = partner_identifier

    def __repr__(self):
        return "<MtCandidateObject | position: %s, orientation: %s, id: %s, partner id: %s> \n" %\
        (self.position, self.orientation, self.identifier, self.partner_identifier)


def fit_ellipse(cc, verbose=False, plot=False):
    """
    Fits an ellipse to the connected components of a binary image and returns major, minor axis, angle to x axis 
    as well as the centroid of each connected component.
    
    Parameters:
    ---------------------
    cc: Labeled input image. I.e. cc contains a matrix where connected components are indicated by different values.
        This is readily available as output from ndimage.label(). Labels with value zero are ignored.

    
    Returns:
    --------------------
    A list of major axis, minor axis, angle to x axis x_centroid, y_centroid for each labeled component in the input.
    """

    props = skimage.measure.regionprops(cc, cache=True)
    cc_features = []
    for prop in props:
        a = prop.major_axis_length
        b = prop.minor_axis_length
        phi = prop.orientation
        y_0, x_0 = prop.centroid 
        # NOTE: Careful with x_0, y_0 -> x <-> columns, y <-> rows
        
        cc_features.append((a,b,phi,x_0,y_0))
        
        if verbose:  
            print "centroid: (", x_0, ", ", y_0, ")\n"     
            print "major_axis_length:", prop.major_axis_length, '\n'
            print "minor_axis_length:", prop.minor_axis_length, '\n'     
            print 'orientation:', prop.orientation, '\n'
            print "inertia_tensor_eigenvals:", prop.inertia_tensor_eigvals, '\n'
        
        if plot:
            plt.imshow(cc)
            plt.scatter(x_0, y_0, s=10, color="black")
            plot_ellipse(a, b, phi, x_0, y_0)

    return cc_features



def plot_ellipse(a, b, alpha, x_0, y_0):
    theta_grid = np.arange(0, 2*np.pi, 0.1)
    ellipse_x_r = a/2.0 * np.cos(theta_grid)
    ellipse_y_r = b/2.0 * np.sin(theta_grid)
    #NOTE Division by 2! Needed because output of prop is the diameter.
 
    #NOTE: Correction applied only for plotting. Needed for correct behavior of R.
    if alpha<0: 
        alpha = alpha + np.pi

    R = np.array(([np.cos(alpha),np.sin(alpha)],[-np.sin(alpha), np.cos(alpha)]))
    r_ellipse = np.dot(R,np.array([ellipse_x_r, ellipse_y_r]))
    plt.plot(r_ellipse[0,:] + x_0, r_ellipse[1,:] + y_0, linewidth=0.8, color="black")
    #plt.show()

def f_delta(d, diam_out):
    """
    Calculates the relation between length of a microtubule and the angle between the x-y plane of that microtubule given
    the section thickness in z direction and the outer diamater of a microtubule.

    Parameters:
    ----------------
    d = section thickness of the data in nm.
    
    diam_out = outer diameter of microtubule in nm. Usually assumed to be 24 nm. See literature.

    
    Returns:
    ----------------
    Lambda function that encodes the relationship between the length of a mt (x) and the angle to the x-y plane.
    """

    # NOTE: We only operate in +,+ quadrant. So arctan should be fine.
    return lambda x: np.arcsin(diam_out/np.sqrt(x**2 + d**2)) + np.arctan(d/x) 


def get_orientation(cc, voxel_x, voxel_y, voxel_z, length_correction, verbose=False, plot_ellipses=False):
    """
    Returns the angles of connected components in labeled binary image w.r.t z and xy-plane as well as center positions,
    length in pixel and the physical length (i.e. scaled by voxel size)

    
    
    Assumptions: 
    -----------------
    We assyme approximate isotropy in x-y plane. However this can be extended to the non-isotropic case. In the current 
    implementation this would lead to an error to the calculated value of the physical length of a mt which in turn would
    give us an error on the angle to the z axis (delta).

    Additionally we assume a microtubule is perpendicular to the xy plane if its length (as estimated by major principle axis 
    of the fitted ellipse) is smaller than 24 nm. However, the length estimate is calibrated (see later).
    

    
    Parameter:
    ---------------
    cc: Labeled input image. I.e. cc contains a matrix where connected components are indicated by different values.
        This is readily available as output from ndimage.label(). Labels with value zero are ignored.
 
    voxel_(xyz): dimensions of the voxels in nm.

    length_correction: The deviation from 24 nm when calculating the length of mt's in a dataset with only perpendicular mt's.
                       Can be set to zero for default behavior.


    
    Returns:
    ---------------
    5 lists containing the angle to the xy plane, to the xz plane, 
    the centroid, the physical length, the pixel length for each labeled connected
    component in the input image.
    """



    cc_features = fit_ellipse(cc, verbose,  plot_ellipses)
    assert voxel_x == voxel_y  
    """
    Otherwise the length of l is wrong. 
    Need isotropy here. Needs to be in nm.
    """
    angle_xy = []
    angle_xz = []
    length_physical = []
    length_pixel = []
    centroid = []

    d = voxel_z
    f = f_delta(voxel_z, diam_out)
        
    for feature in cc_features:

        l = (feature[0] + length_correction) * voxel_x 
        assert l>= 0 
        """
        Otherwise we might get in 
        trouble with ambiguities from arctan()
        """

        gamma = feature[2]
        """
        in interval [- pi/2, + pi/2], 
        where - indicates nw/se direction and + ne/sw direction
        """
        if l <= diam_out:
            delta=np.pi/2 
        """
        In case the length is smaller or equal to the 
        diameter of a mt we assume its perp. 
        NOTE: THIS NEEDS TO BE CONSISTENT WITH POLARITY NONE TO GET PI FOR THETA!
        """
        else:
            delta = f(l) 
            # angle delta -> see sketches, NOTE: NEED TO HANDLE ZERO l!
        
        angle_xy.append(gamma)
        angle_xz.append(delta)
        length_physical.append(l)
        length_pixel.append(feature[0]+length_correction)
        centroid.append((feature[3], feature[4]))

    return angle_xy, angle_xz, centroid, length_physical, length_pixel



def get_vector(pos_plus, angle_xz, pos_minus=None, verbose=False, plot=False):
    """

    """
    pos_plus = np.array(pos_plus)
    if pos_minus is None: # No plus minus in this case, i.e. perpendicular
        phi=0.0 # arbitrary as here the vector goes down perp.
        theta=np.pi
        r = 1.0
        
        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(theta)

        return (phi, theta, r), (x, y, z)
    
    else:
        pos_minus = np.array(pos_minus) 
        theta = angle_xz + np.pi/2.0
        
        from_m_to_p = pos_plus - pos_minus
        from_p_to_m = pos_minus - pos_plus

        phi_m_to_p = np.arctan2(from_m_to_p[1],from_m_to_p[0])
        phi_p_to_m = np.arctan2(from_p_to_m[1],from_p_to_m[0])
                
        
        r = 1./np.sin(angle_xz)

        x_m_to_p = r * np.sin(theta) * np.cos(phi_m_to_p)
        y_m_to_p = r * np.sin(theta) * np.sin(phi_m_to_p)
        z_m_to_p = r * np.cos(theta)

        x_p_to_m = r * np.sin(theta) * np.cos(phi_p_to_m)
        y_p_to_m = r * np.sin(theta) * np.sin(phi_p_to_m)
        z_p_to_m = r * np.cos(theta)

        if verbose:        
            print "vec(plus_to_minus)", x_p_to_m, y_p_to_m, z_p_to_m
            print "vec(mimus_to_plus)", x_m_to_p, y_m_to_p, z_m_to_p
        
        if plot:
            plt.figure()
            plt.plot([0, x_p_to_m], [0, y_p_to_m], color="red")
            plt.plot([0, x_m_to_p], [0, y_m_to_p], color="green")
            plt.xlim([-2, 2])
            plt.ylim([-2, 2])
            plt.axvline(x=0)
            plt.axhline(y=0)
            plt.show()

        return (phi_m_to_p, theta, r), (phi_p_to_m, theta, r), (x_m_to_p, y_m_to_p, z_m_to_p), (x_p_to_m, y_p_to_m, z_p_to_m)



def get_candidates(cc, 
                   slice_number, 
                   voxel_x, 
                   voxel_y, 
                   voxel_z, 
                   length_correction, 
                   verbose=False, 
                   plot=False, 
                   identifier_0=0):
    
    list_angle_xy, list_angle_xz, list_centroid, list_length_physical, list_length_pixel =\
        get_orientation(cc, voxel_x, voxel_y, voxel_z, length_correction=length_correction) #PERP CASE HANDLED IN HERE
    
    if plot:
        features = fit_ellipse(cc)
    
    candidate_list = []
    j = 0
    identifier = identifier_0
    
    for angle_xy, angle_xz, centroid, length_physical, length_pixel in\
        zip(list_angle_xy, list_angle_xz, list_centroid, list_length_physical, list_length_pixel):

        if length_physical <= diam_out:
            spherical, cartesian = get_vector(centroid, angle_xz) # NOTE: Maybe normalize vector here
            candidate = MtCandidate(position=centroid + (slice_number - 0.5,), 
                        orientation=cartesian, identifier=identifier)
            candidate_list.append(candidate)
            identifier += 1

        else:
            x_0 = centroid[0]
            y_0 = centroid[1]
            
            if angle_xy < 0:
                correction_x = (length_pixel/2.0)*np.cos(angle_xy)
                correction_y = -(length_pixel/2.0)*np.sin(angle_xy)

                x_plus = x_0 + correction_x
                y_plus = y_0 + correction_y
                
                x_minus = x_0 - correction_x
                y_minus = y_0 - correction_y
                
            else:
                correction_x = (length_pixel/2.0)*np.cos(angle_xy)
                correction_y = (length_pixel/2.0)*np.sin(angle_xy)

                x_plus = x_0 - correction_x
                y_plus = y_0 + correction_y
    
                x_minus = x_0 + correction_x
                y_minus = y_0 - correction_y
                
            
                            
            if verbose:
                print "\n-------------------------------------------------"
                print "Polarity Correction: \n"
                print 'Centroid: (x_0, y_0):', x_0, y_0, '\n'
                print "x_correction: ", correction_x, '\n'
                print "y_correction: ", correction_y, '\n'
                print "angle_xy: ", angle_xy, '\n'
                print "phi: ", spherical_plus[0], '\n'
                print "--------------------------------------------------\n"
                print "x_plus, y_plus:", x_plus, y_plus, "\n"
                print "x_minus, y_minus:", x_minus, y_minus, "\n"
                print "--------------------------------------------------\n"
            
            position_plus = (x_plus, y_plus) + (slice_number,)
            position_minus = (x_minus, y_minus) + (slice_number,)

            spherical_minus, spherical_plus, cartesian_minus, cartesian_plus =\
                get_vector((x_plus, y_plus), angle_xz, (x_minus, y_minus))
            
            if plot:
                plt.figure()
                plt.scatter(cartesian_plus[0] + x_plus, cartesian_plus[1] + y_plus, color="green", s=10)
                plt.scatter(cartesian_minus[0] + x_minus , cartesian_minus[1] + y_minus, color="red", s=10)
                plt.scatter(x_plus, y_plus, color="green", s=10)
                plt.scatter(x_minus, y_minus, color="red", s=10)
                plt.imshow(cc, origin="lower")
                plot_ellipse(*features[j])
                plt.show()

            #Change position of vectors to lie in the middle of the section at the same point:
            position_plus = (x_0, y_0) + (slice_number - 0.5,)
            position_minus = position_plus

            candidate_plus = MtCandidate(position=position_plus, 
                                         orientation=cartesian_plus, 
                                         identifier=identifier, 
                                         partner_identifier=identifier+1)
            identifier += 1            

            candidate_minus = MtCandidate(position=position_minus, 
                                          orientation=cartesian_minus, 
                                          identifier=identifier, 
                                          partner_identifier=identifier-1)
            identifier += 1

            candidate_list.append(candidate_plus)
            candidate_list.append(candidate_minus)

        j += 1

    max_identifier = identifier

    return candidate_list, max_identifier



def get_toy_cc(length, width, angle):
    cc_toy_canvas = np.zeros((2* length, 2*width), dtype=int)
    
    for i in xrange(length):
        for j in xrange(width):
            cc_toy_canvas[length + i][width + j] = 1
    
    cc_toy_canvas = ndimage.rotate(cc_toy_canvas, angle, reshape=True)
    return cc_toy_canvas



def get_length_correction(input_file, threshold, gaussian_sigma, voxel_x):
    """
    The parameters here should be set exactly as in the data set to analyze! The input data set should show perp. mts!
    """
    
    cc_list, n_list = threshold_prob_map(threshold=threshold, inputfile=inputfile, gaussian_sigma=gaussian_sigma, plot=False)
    l_sum = 0
    n = 0

    for cc in cc_list:
        cc_features = fit_ellipse(cc)
        
        for feature in cc_features:
            l = feature[0] * voxel_x
            assert l>= 0 # Otherwise we might get in trouble with ambiguities from arctan()
            l_sum += l
            n += 1

    return (diam_out - (l_sum/float(n)))/voxel_x


def test_angle_estimate(voxel_x=5.0, voxel_y=5.0, voxel_z=50.0, plot=True):
    
    length_list = (1, 2, 5, 10, 20)
    width_list = (1, 2, 5, 10, 20) 
    angle_list = (0, 0.14*np.pi, np.pi/4., np.pi/2., np.pi)

    perp_length_pixel = diam_out/voxel_x
    print "Perp Length Pixel: ", perp_length_pixel, "\n"
    
    toy_cc_list = []
    param_list = []

    for length in length_list:
        for width in width_list: 
            for angle in angle_list:
                toy_cc_list.append(get_toy_cc(length=length, width=width, angle=angle))
                param_list.append((length, width, angle))

    candidate_list = []

    layer_ = 0
    identifier_0 = 0

    for cc in toy_cc_list:
        candidate, max_identifier = get_candidates(cc, 
                                                   layer_, 
                                                   voxel_x, 
                                                   voxel_y, 
                                                   voxel_z, 
                                                   length_correction=0.0, 
                                                   plot=plot, 
                                                   identifier_0=identifier_0)
        identifier_0 = max_identifier
        candidate_list.extend(candidate)
        layer_ += 1

    for candidate in candidate_list:
        print "Position: ", candidate.position
        print "Orientation: ", candidate.orientation
        print "Identifier: ", candidate.identifier
        print "Partner Identifier: ", candidate.partner_identifier
        print "\n \n"
