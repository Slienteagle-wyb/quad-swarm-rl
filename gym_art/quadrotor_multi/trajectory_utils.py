import copy
import numpy as np
from gym_art.quadrotor_multi.quad_types import *
from scipy.spatial.transform import Rotation

R_RANGE = (3.0, 5.0)  # random track with a length from 110 to 150 meters containing 10 gates
CAME_FOV = 85.0


def random_sample(value_range):
    return (value_range[1] - value_range[0])*np.random.random() + value_range[0]


def polar_translation(r, theta, phi):
    x = r * np.cos(theta) * np.sin(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(phi)
    return Vector3r(x, y, z)


def cartesian_translation(x, y, z):
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(z / r)
    phi = np.arctan2(y, x)
    return Vector3r(r, theta, phi)


def convert_body_2_world(body, q_o_b):
    rotation = Rotation.from_quat(q_o_b.to_numpy_array())
    world_np = rotation.apply(body.to_numpy_array())
    world = Vector3r(world_np[0], world_np[1], world_np[2])
    return world


def get_yaw_base(p_o_b):
    q_o_b = p_o_b.orientation
    rotation = Rotation.from_quat([q_o_b.x_val, q_o_b.y_val, q_o_b.z_val, q_o_b.w_val])
    euler_angles = rotation.as_euler('ZYX')
    return euler_angles[0]


def included_angle(vec_1, vec_2):
    dot_product = np.dot(vec_1, vec_2)
    cos_theta = dot_product / (np.linalg.norm(vec_1) * np.linalg.norm(vec_2))
    return np.arccos(np.abs(cos_theta))


def generate_random_track(formation_center, num_gates):
    gate_poses = []
    gate_poses_spherical = []
    origin_pos = Vector3r(formation_center[0], formation_center[1], formation_center[2])
    origin_pose = Pose(origin_pos, Quaternionr(0, 0, 0.707, 0.707))  # rpy (0, 0, pi/2)
    gate_poses.append(origin_pose)
    # generate an extra final gate pos for hovering
    for i in range(num_gates + 1):
        # # set the random position of gate in spherical coordinates
        r = random_sample(R_RANGE)
        # set theta range inside the camera FOV
        theta_range = (-CAME_FOV/3.0*np.pi/180.0, CAME_FOV/3.0*np.pi/180.0)
        theta = random_sample(theta_range)
        phi_prime = np.arcsin(0.6 / r)  # the trajectories with elevation change up to 17m (each 1.7m)
        phi_range = (-phi_prime, phi_prime)
        phi = random_sample(phi_range) + np.pi/2.0
        if i == 0:
            theta = 0.0
            phi = np.pi / 2.0
        gate_pos_body = polar_translation(r, theta, phi)
        gate_pos_world = convert_body_2_world(gate_pos_body, gate_poses[-1].orientation)
        gate_pos_world = gate_poses[-1].position + gate_pos_world
        # # set the random yaw of gate
        psi_range = (-np.pi/6.0, np.pi/6.0)
        psi = random_sample(psi_range)
        if i == 0:
            psi = 0.0
        yaw_base = get_yaw_base(gate_poses[-1])
        gate_yaw_world = yaw_base + psi
        gate_rot = Rotation.from_euler('ZYX', [gate_yaw_world, 0, 0]).as_quat()
        # # prepare the gate pose in world frame for rendering
        gate_pose = Pose(gate_pos_world, Quaternionr(gate_rot[0], gate_rot[1], gate_rot[2], gate_rot[3]))
        gate_poses.append(gate_pose)

        # # prepare the gate pose in gate's body frame for control using the spherical coordinates
        gate_poses_spherical.append(np.array((r, theta, phi, psi)))

    return gate_poses[1:], gate_poses_spherical[1:]
