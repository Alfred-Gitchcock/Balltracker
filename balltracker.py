import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.ndimage import center_of_mass


def Trajectory(start_pos, start_vel, t):
    """
    Calculates the 3D position of a ball at time t, including gravity acting in -z direction.
    Each coordinate is multiplied by 10 for better discretisation, showing distance in dm istead of m.
    Parameters:
    - start_pos: Tuple or list of 3 elements (x0, y0, z0), should be in m
    - velocity: Tuple or list of 3 elements (vx, vy, vz), should be in m/s
    - t: Time in seconds (float)

    Returns:
    - Tuple (x, y, z): Position at time t
    """

    # x and y are very straightforward
    x = 10 * np.min([start_pos[0] + start_vel[0] * t, 50])
    y = 10 * np.min([start_pos[1] + start_vel[1] * t, 65])

    # modelling z as a dampened harmonic oscillator
    def max_height(z,vz):
        """Calculates maximum height the ball can reach."""
        g = 9.81
        t = np.linspace(0,10,100)
        return np.max(z + vz * t - 0.5 * g * t**2)

    z_max = max_height(start_pos[2], start_vel[2])
    dampening_coeff = 0.5
    angular_freq = 8
    z = 10 * z_max * np.exp(-dampening_coeff * t) * np.abs(np.sin(angular_freq * t))

    return np.round([x, y, z],2)

def insert_ball(frame, position, radius, color):

    """
    Inserts a circle of given radius into input frame at position
    :param frame: np.ndarray of with 2 dimensions of arbitrary size
    :param position: Tuple or list of 2 elements (x, y), coordinates at which to center the circle
    :param radius: Int, radius of circle
    :param color: Int, color of circle
    :return: frame: np.ndarray, same as input array but now with added ball
    """

    yy, xx = np.ogrid[:frame.shape[0], :frame.shape[1]]
    ball = (xx - position[0])**2 + (yy - position[1])**2 <= radius**2
    frame[ball] = color
    return frame

def Image_Generator(ball_pos):

    """
    Generates images as np.ndarrays for two cameras (two different povs of the same space) and puts a ball in them
    :param ball_pos:
    :return:
    """

    dims_A = (350,650)
    dims_B = (350,500)

    # creating image frames of cam A (at the goal) and cam B (on the side)
    cam_A = np.zeros(dims_A)
    cam_B = np.zeros(dims_B)

    # get projected positions for each camera frame
    pos_A = [ball_pos[1], ball_pos[2]] # cam A sees y- and z-projections
    pos_B = [ball_pos[0], ball_pos[2]] # cam B sees x- and z-projections

    # put the ball into the frames
    ballframe_A = insert_ball(cam_A,pos_A,8,255)
    ballframe_B = insert_ball(cam_B, pos_B, 8, 255)

    return ballframe_A, ballframe_B

def Ballfinder(frames, threshold):

    """
    Detects a ball in a given set of input frames via scipy's center_of_mass function
    :param frames: List or numpy.ndarray containing the frames to be analysed
    :param threshold: Int, detection threshold
    :return: List containing detected ball-position for each frame
    """

    tracked_positions = []

    for frame in frames:
        binary = frame > threshold
        if np.any(binary):
            cy, cx = center_of_mass(binary)  # returns (row, col) = (y, x)
            tracked_positions.append((int(cx), int(cy)))
        else:
            tracked_positions.append(None)

    return tracked_positions

def Tracker(cam_A, cam_B, timesteps):

    """
    Returns [t, x, y, z] coordinates of a ball moving through images from cams A and B.
    :param cam_A: List or np.ndarray containing frames (np.ndarrays) of camera A.
    :param cam_B: List or np.ndarray containing frames (np.ndarrays) of camera B.
    :param timesteps: List or np.ndarray containing the timesteps at which the camera frames were recorded.
    :return: pos_4D: np.ndarray containing the 4D position of the ball in the form [t, x, y, z]
    """

    # getting tracked positions
    pos_A = Ballfinder(cam_A, 128) # y and z coordinates
    pos_B = Ballfinder(cam_B, 128) # x and z coordinates

    # initialising x, y and z arrays to store coordinates
    x = np.zeros_like(timesteps)
    y = np.zeros_like(timesteps)
    z = np.zeros_like(timesteps)

    for i in range(len(timesteps)):
        x[i] = pos_B[i][0]
        y[i] = pos_A[i][0]
        z[i] = pos_A[i][1] # z is the same for both cameras, can choose either one here

    pos_4D = np.array([timesteps, x, y, z])

    return pos_4D


# Example Usage:

# initial values and timesteps at which pictures are taken
pos = [0,10,0]
vel = [10, 8, 15]
times = [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]

cam_A = []
cam_B = []

for t in times:
    ball_pos = Trajectory(pos, vel, t)
    frame_A, frame_B = Image_Generator(ball_pos)
    cam_A.append(frame_A)
    cam_B.append(frame_B)

tracked_positions = Tracker(cam_A, cam_B, times)

print(tracked_positions)

# Verification
fig, axes = plt.subplots(nrows=7, ncols=2, figsize=(7, 14))

for i in range(7):

    t = tracked_positions[0][i]
    x = tracked_positions[1][i]
    y = tracked_positions[2][i]
    z = tracked_positions[3][i]

    center_A = (y,z)
    center_B = (x,z)

    circle_A = patches.Circle(center_A, 15, edgecolor='red', facecolor='none', linewidth=2)
    circle_B = patches.Circle(center_B, 15, edgecolor='red', facecolor='none', linewidth=2)

    axes[i, 0].imshow(cam_A[i], cmap='gray')
    axes[i, 0].add_patch(circle_A)
    axes[i, 0].set_ylim([0, 150])
    axes[i, 0].set_xlim([0, 650])
    axes[i, 0].set_title(f'Cam A frame at t = {t}s')

    axes[i, 1].imshow(cam_B[i], cmap='gray')
    axes[i, 1].add_patch(circle_B)
    axes[i, 1].set_ylim([0, 150])
    axes[i, 1].set_xlim([0, 500])
    axes[i, 1].set_title(f'Cam B frame at t = {t}s')

plt.tight_layout()
plt.show()