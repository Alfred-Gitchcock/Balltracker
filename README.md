# Balltracker

User Story:

The trainers at Wien Rapid wish to generate statistics on how accurate their player can kick a ball, so they know who to put forward and when. To do this they have placed two cameras on the field. One built into the cross beam of the goals, and the other on the side of the field looking at the goal square. 

They need two things:

- a generator script that creates a stack of images with a ball moving through it. This is to create test data for the following function.

- a python function that can track a ball (a circle) as it moves through a stack of images (cube), and returns the t,x,y,z coordinates of the ball

Assumptions about general setup:
- Cartesian coordinate system with origin on the intersection between the middle line and the left (from cam A's PoV) border of the field.
  - Only the part of the field between the goal on which Cam A is mounted and the middle line is taken into account, as this is assumed to be the half of the playing field in which Rapid's players are on the offensive.
  - Dimensions of simulation domain:
    - x: 0 - 500dm (Half of standard playing field length of around 100m)
    - y: 0 - 650dm (Full width of playing field)
    - z: 0 - 150dm (Ball usually shouldn't fly higher than 15m)
  - x-axis points towards the goal along the field's border
  - y-axis points towards the center of the field
  - z-axis points upwards
- Cam A is mounted on the goal
  - sees y-z plane
- Cam B is mounted to the left (again from Cam A's PoV) of the field
  - sees x-z plane