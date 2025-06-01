import numpy as np
import matplotlib.pyplot as plt
import numba
import random


def Trajectory(start_pos, start_vel, t):
    """
    Calculates the 3D position of a ball at time t, including gravity acting in -z direction.

    Parameters:
    - start_pos: Tuple or list of 3 elements (x0, y0, z0)
    - velocity: Tuple or list of 3 elements (vx, vy, vz)
    - t: Time in seconds (float)

    Returns:
    - Tuple (x, y, z): Position at time t
    """

    if len(start_pos) != 3 or len(start_vel) != 3:
        raise ValueError("start_pos and velocity must be 3-element vectors.")

    g = 9.81

    x = start_pos[0] + start_vel[0] * t
    y = start_pos[1] + start_vel[1] * t
    z = start_pos[2] + start_vel[2] * t - 0.5 * g * t ** 2  # gravity affects z

    return (x, y, z)

# generate random noise for input image x
@numba.njit
def RandomNoise(x):
    x = x.reshape(-1) # flat view
    for i in range(len(x)):
        x[i] += random.random()

def Image_Generator(N, ball_pos, noise = True):

    # creating image frames of cam A (at the goal) and cam B (on the side)
    cam_A = np.zeros((N,N))
    cam_B = np.zeros((N,N))

    # filling images with random noise
    if noise == True:
        cam_A = RandomNoise(cam_A)
        cam_B = RandomNoise(cam_A)
    else:
        pass

    # put the ball into the images
    pos_A = [ball_pos[1], ball_pos[2]] # cam A sees y- and z-projections
    pos_B = [ball_pos[0], ball_pos[2]] # cam B sees x- and z-projections

    # CONTINUE WORK HERE