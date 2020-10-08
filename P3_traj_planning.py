import numpy as np
from P1_astar import DetOccupancyGrid2D, AStar
from P2_rrt import *
import scipy.interpolate
import matplotlib.pyplot as plt
from HW1.P1_differential_flatness import *
from HW1.P2_pose_stabilization import *
from HW1.P3_trajectory_tracking import *

class SwitchingController(object):
    """
    Uses one controller to initially track a trajectory, then switches to a
    second controller to regulate to the final goal.
    """
    def __init__(self, traj_controller, pose_controller, t_before_switch):
        self.traj_controller = traj_controller
        self.pose_controller = pose_controller
        self.t_before_switch = t_before_switch

    def compute_control(self, x, y, th, t):
        """
        Inputs:
            (x,y,th): Current state
            t: Current time

        Outputs:
            V, om: Control actions
        """
        ########## Code starts here ##########
        if t > self.t_before_switch:
            return self.traj_controller.compute_control(x, y, th, t)
        else:
            return self.pose_controller.compute_control(x, y, th, t)
        ########## Code ends here ##########

def compute_smoothed_traj(path, V_des, alpha, dt):
    """
    Fit cubic spline to a path and generate a resulting trajectory for our
    wheeled robot.

    Inputs:
        path (np.array [N,2]): Initial path
        V_des (float): Desired nominal velocity, used as a heuristic to assign nominal
            times to points in the initial path
        alpha (float): Smoothing parameter (see documentation for
            scipy.interpolate.splrep)
        dt (float): Timestep used in final smooth trajectory
    Outputs:
        traj_smoothed (np.array [N,7]): Smoothed trajectory
        t_smoothed (np.array [N]): Associated trajectory times
    Hint: Use splrep and splev from scipy.interpolate
    """
    ########## Code starts here ##########
    x = np.array([point[0] for point in path])
    y = np.array([point[1] for point in path])
    #t_old is not right it should be the  current and next point
    t_old = scipy.integrate.cumtrapz([0] + [V_des/np.sqrt((x[i] - x[i+1])**2 + (y[i] - y[i+1])**2) for i in range(0, len(x)-1)], initial = 0)
    print(len(t_old))

    t_smoothed = np.arange(0.0, t_old[-1], dt)
    print(len(t_smoothed))
    traj_coefficients_x= scipy.interpolate.splrep(t_old, x, s = alpha)
    traj_coefficients_y = scipy.interpolate.splrep(t_old, y, s = alpha)
    x_new= scipy.interpolate.splev(t_smoothed, traj_coefficients_x)
    y_new = scipy.interpolate.splev(t_smoothed, traj_coefficients_y)
    xd_new= scipy.interpolate.splev(t_smoothed, traj_coefficients_x, der = 1)
    yd_new = scipy.interpolate.splev(t_smoothed, traj_coefficients_y, der = 1)
    xdd_new= scipy.interpolate.splev(t_smoothed, traj_coefficients_x, der = 2)
    ydd_new = scipy.interpolate.splev(t_smoothed, traj_coefficients_y, der = 2)
    theta = np.arctan2(yd_new, xd_new)
    traj_smoothed = np.array([x_new, y_new, theta, xd_new, yd_new, xdd_new, ydd_new ]).transpose()
    ########## Code ends here ##########

    return traj_smoothed, t_smoothed

def modify_traj_with_limits(traj, t, V_max, om_max, dt):
    """
    Modifies an existing trajectory to satisfy control limits and
    interpolates for desired timestep.

    Inputs:
        traj (np.array [N,7]): original trajecotry
        t (np.array [N]): original trajectory times
        V_max, om_max (float): control limits
        dt (float): desired timestep
    Outputs:
        t_new (np.array [N_new]) new timepoints spaced dt apart
        V_scaled (np.array [N_new])
        om_scaled (np.array [N_new])
        traj_scaled (np.array [N_new, 7]) new rescaled traj at these timepoints
    Hint: This should almost entirely consist of calling functions from Problem Set 1
    """
    ########## Code starts here ##########
    # traj_new = []
    # tau_new = []
    # V_new= []
    # om_new = []
    # for i in range(np.shape(traj)[0]-1):
    #     s_0 = State(x=traj[i, 0], y=traj[i, 1], V=V_max, th=traj[i, 2])
    #     s_f = State(x=traj[i+1, 0], y=traj[i+1, 1], V=V_max, th=traj[i+1, 2])
    #     tf = t[i+1] - t[i]
    #     N = int(tf/dt) + 1
    #     traj_curr, tau_curr, V_tilde_curr, om_tilde_curr = compute_traj_with_limits(s_0, s_f, tf, N, V_max, om_max)
    #     if i == 0:
    #         traj_new = traj_curr
    #         V_new = V_tilde_curr
    #         om_new = om_tilde_curr
    #         tau_new = tau_curr
    #     else:
    #         np.append(traj_new, traj_curr)
    #         np.append(V_new, V_tilde_curr)
    #         np.append(om_new, om_tilde_curr)
    #         np.append(tau_new, tau_curr)
    #
    # t_new, V_scaled, om_scaled, traj_scaled = interpolate_traj(traj_new, tau_new, V_new, om_new, dt, s_f)

    V,om = compute_controls(traj=traj)
    s = compute_arc_length(V, t)
    s_f = State(x=traj[-1, 0], y=traj[-1, 1], V=V_max, th=traj[-1, 2])

    V_tilde = rescale_V(V, om, V_max, om_max)
    tau = compute_tau(V_tilde, s)
    om_tilde = rescale_om(V, om, V_tilde)

    t_new, V_scaled, om_scaled, traj_scaled = interpolate_traj(traj, tau, V_tilde, om_tilde, dt, s_f)
    ########## Code ends here ##########

    return t_new, V_scaled, om_scaled, traj_scaled
