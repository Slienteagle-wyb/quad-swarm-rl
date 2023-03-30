import time
from collections import deque

import numpy as np
from gym_art.quadrotor_multi.trajectory_utils import *
from gym_art.quadrotor_multi.quad_utils import perform_collision_between_drones, perform_collision_with_obstacle, \
    calculate_collision_matrix, calculate_drone_proximity_penalties, calculate_obst_drone_proximity_penalties

from gym_art.quadrotor_multi.quadrotor_multi_obstacles import MultiObstacles
from gym_art.quadrotor_multi.quadrotor_single import GRAV, QuadrotorSingle
from gym_art.quadrotor_multi.quadrotor_multi_visualization import Quadrotor3DSceneMulti
from gym_art.quadrotor_multi.quad_scenarios import create_scenario
from gym_art.quadrotor_multi.quadrotor_control import *


EPS = 1E-6


class QuadrotorEnvRacing(gym.Env):
    def __init__(self,
                 dynamics_params='DefaultQuad', dynamics_change=None,
                 dynamics_randomize_every=None, dyn_sampler_1=None, dyn_sampler_2=None,
                 raw_control=True, raw_control_zero_middle=True, dim_mode='3D', tf_control=False, sim_freq=200.,
                 sim_steps=2, obs_repr='xyz_vxyz_acc_R_omega', ep_time=7, obstacle_num=0, room_length=10, room_width=10,
                 room_height=10, init_random_state=False, rew_coeff=None, sense_noise='default', verbose=False,
                 gravity=GRAV,
                 resample_goals=False, t2w_std=0.005, t2t_std=0.0005, excite=False, dynamics_simplification=False,
                 quads_mode='traverse_gate', quads_formation='circle_horizontal', quads_formation_size=-1.0,
                 quads_use_numba=False, quads_settle=False, quads_settle_range_meters=1.0,
                 quads_vel_reward_out_range=0.8, quads_obstacle_mode='static', quads_view_mode='local',
                 quads_obstacle_type='sphere', quads_obstacle_size=0.5, collision_force=True,
                 adaptive_env=False, obstacle_traj='static', collision_hitbox_radius=2.0,
                 collision_falloff_radius=2.0, collision_smooth_max_penalty=10.0,
                 use_replay_buffer=False, controller_type=None,
                 obstacle_obs_mode='relative', obst_penalty_fall_off=10.0, vis_acc_arrows=False,
                 viz_traces=25, viz_trace_nth_step=1, track_gate_nums=10, num_vis_gates=3, num_render_gates=10, ):

        super().__init__()
        self.room_dims = (room_length, room_width, room_height)
        self.adaptive_env = adaptive_env  # default is false
        self.quads_view_mode = quads_view_mode

        self.preceding_goals = None
        self.goal_central = np.array([0., 0., 2.])
        self.obstacle_num = obstacle_num
        self.track_gate_nums = track_gate_nums
        self.env = QuadrotorSingle(
            dynamics_params, dynamics_change, dynamics_randomize_every, dyn_sampler_1, dyn_sampler_2,
            raw_control, raw_control_zero_middle, dim_mode, tf_control, sim_freq, sim_steps,
            obs_repr, ep_time, obstacle_num, room_length, room_width, room_height, init_random_state,
            rew_coeff, sense_noise, verbose, gravity, t2w_std, t2t_std, excite, dynamics_simplification,
            quads_use_numba, quads_settle, quads_settle_range_meters,
            quads_vel_reward_out_range, quads_view_mode, quads_obstacle_mode, obstacle_num,
            controller_type)

        self.resample_goals = resample_goals

        # we don't actually create a scene object unless we want to render stuff
        self.scene = None  # manipulate the 3d render assets
        self.num_vis_gates = num_vis_gates
        self.num_render_gates = num_render_gates

        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

        # reward shaping
        self.rew_coeff = dict(
            progress=1.0, safety=0.1, spin=0.1, effort=0.05, orient=1.0,
            pos=0., action_change=0., yaw=0., rot=0., attitude=0., vel=0.,
            quadcol_bin=0., quadcol_bin_smooth_max=0., precede=0.0, crash=0.,
            quadsettle=0., quadcol_bin_obst=0., quadcol_bin_obst_smooth_max=0.0
        )
        rew_coeff_orig = copy.deepcopy(self.rew_coeff)

        if rew_coeff is not None:
            assert isinstance(rew_coeff, dict)
            assert set(rew_coeff.keys()).issubset(set(self.rew_coeff.keys()))
            self.rew_coeff.update(rew_coeff)
        for key in self.rew_coeff.keys():
            self.rew_coeff[key] = float(self.rew_coeff[key])

        orig_keys = list(rew_coeff_orig.keys())
        # Checking to make sure we didn't provide some false rew_coeffs (for example by misspelling one of the params)
        assert np.all([key in orig_keys for key in self.rew_coeff.keys()])

        # Aux variables for observation space of quads
        self.quad_arm = self.env.dynamics.arm
        self.control_freq = self.env.control_freq
        self.control_dt = 1.0 / self.control_freq
        self.drone_pos = np.zeros(3)  # Matrix containing all positions
        self.quads_mode = quads_mode
        if obs_repr == 'xyz_vxyz_R_omega':
            self.obs_drone_size = 18
        elif obs_repr == 'xyz_vxyz_R_omega_wall':
            self.obs_drone_size = 24
        elif obs_repr == 'xyz_vxyz_acc_R_omega':
            self.obs_drone_size = 21
        else:
            raise NotImplementedError(f'{obs_repr} not supported!')

        # Aux variables for rewards
        self.rews_settle = np.zeros(1)
        self.rews_settle_raw = np.zeros(1)

        # Aux variables for scenario
        # scenario manipulate the track setting and the quadrotor dynamics
        self.scenario = create_scenario(quads_mode=quads_mode, envs=(self.env,), num_agents=1,
                                        room_dims=self.room_dims, room_dims_callback=self.set_room_dims,
                                        rew_coeff=self.rew_coeff,
                                        quads_formation=quads_formation, quads_formation_size=quads_formation_size,
                                        track_gate_nums=self.track_gate_nums, )
        self.quads_formation_size = quads_formation_size

        # Set Obstacles as the gates of racing track.
        self.multi_obstacles = None
        self.obstacle_mode = quads_obstacle_mode
        self.use_obstacles = self.obstacle_mode != 'no_obstacles' and self.obstacle_num > 0
        if self.use_obstacles:
            obstacle_max_init_vel = 4.0 * self.env.max_init_vel
            obstacle_init_box = self.env.box  # box of env is: 2 meters
            # This parameter is used to judge whether obstacles are out of room, and then, we can reset the obstacles
            self.track_room = self.room_dims  # [[-5, -5, 0], [5, 5, 10]]
            dt = 1.0 / sim_freq
            self.set_obstacles = np.zeros(self.obstacle_num, dtype=bool)
            self.obstacle_shape = quads_obstacle_type
            self.obst_penalty_fall_off = obst_penalty_fall_off
            self.multi_obstacles = MultiObstacles(
                mode=self.obstacle_mode, num_obstacles=self.obstacle_num, max_init_vel=obstacle_max_init_vel,
                init_box=obstacle_init_box, dt=dt, quad_size=self.quad_arm, shape=self.obstacle_shape,
                size=quads_obstacle_size, traj=obstacle_traj, obs_mode=obstacle_obs_mode
            )

            # collisions between gates and quadrotors
            self.obst_quad_collisions_per_episode = 0
            self.prev_obst_quad_collisions = []

        # set render
        self.simulation_start_time = 0
        self.frames_since_last_render = self.render_skip_frames = 0
        self.render_every_nth_frame = 1
        self.render_speed = 1.0  # set to below 1 slow mode, higher than 1 for faster forward inference

        # measuring the total number of pairwise collisions per episode
        self.collisions_per_episode = 0

        # some collisions may happen because the quadrotors get initialized on the collision course
        # if we wait a couple of seconds, then we can eliminate all the collisions that happen due to initialization
        # this is the actual metric that we want to minimize
        self.collisions_after_settle = 0
        self.collisions_grace_period_seconds = 1.5

        # collision proximity penalties
        self.collision_hitbox_radius = collision_hitbox_radius
        self.collision_falloff_radius = collision_falloff_radius
        self.collision_smooth_max_penalty = collision_smooth_max_penalty

        self.prev_drone_collisions, self.curr_drone_collisions = [], []
        self.all_collisions = {}
        self.apply_collision_force = collision_force

        # set to true whenever we need to reset the OpenGL scene in render()
        self.reset_scene = False
        self.vis_acc_arrows = vis_acc_arrows
        self.viz_traces = viz_traces
        self.viz_trace_nth_step = viz_trace_nth_step

        self.last_step_unique_collisions = False
        self.crashes_in_recent_episodes = deque([], maxlen=100)
        self.crashes_last_episode = 0

    def set_room_dims(self, dims):
        # dims is a (x, y, z) tuple
        self.room_dims = dims

    def all_dynamics(self):
        return [self.env.dynamics, ]

    def add_preceding_gates_obs(self, drone_obs):
        curr_tracking_gate = self.scenario.gate_poses[self.scenario.curr_gate_ptr]
        relative_pos = curr_tracking_gate.position.to_numpy_array() - self.drone_pos
        polar_drone_pos = cartesian_translation(relative_pos[0], relative_pos[1], relative_pos[2])
        # calculate the included angle between the relative pos vector and gate's normal in \
        # the gate's body frame, we always set the gate face toward the Y axis.
        relative_yaw = included_angle(relative_pos, np.array([0, 1, 0]))
        relative_drone_state = np.concatenate((polar_drone_pos.to_numpy_array(), [relative_yaw]))
        obs_dyna = np.concatenate((relative_drone_state, drone_obs[3:]))

        end_idx = np.minimum(self.scenario.curr_gate_ptr + self.num_vis_gates, self.track_gate_nums)
        visible_gates = np.zeros((self.num_vis_gates, 4))
        if end_idx == self.track_gate_nums:
            visible_gates[0] = self.preceding_goals[-1]
            visible_gates[1] = self.preceding_goals[-1]
            visible_gates = np.concatenate(visible_gates)
        else:
            visible_gates = self.preceding_goals[self.scenario.curr_gate_ptr:end_idx]
            visible_gates = np.concatenate(visible_gates)
        obs = np.concatenate((obs_dyna, visible_gates))
        return obs

    def init_scene_multi(self):
        models = [self.env.dynamics.model, ]  # just one model
        self.scene = Quadrotor3DSceneMulti(
            w=640, h=480, models=models, resizable=True, multi_obstacles=self.multi_obstacles,
            viewpoint=self.env.viewpoint,
            obstacle_mode=self.obstacle_mode, room_dims=self.room_dims, num_agents=1,
            render_speed=self.render_speed, formation_size=self.quads_formation_size,
            vis_acc_arrows=self.vis_acc_arrows, viz_traces=self.viz_traces, viz_trace_nth_step=self.viz_trace_nth_step,
            racing_mode=True, vis_num_gates=self.num_render_gates,
        )

    def reset(self, *args):
        self.scenario.reset()
        self.quads_formation_size = self.scenario.formation_size
        self.goal_central = np.mean(self.scenario.goals, axis=0)

        if self.adaptive_env:
            # TODO: introduce logic to choose the new room dims i.e. based on statistics from last N episodes, etc
            # e.g. self.room_dims = ....
            new_length, new_width, new_height = np.random.randint(1, 31, 3)
            self.room_dims = (new_length, new_width, new_height)

        self.env.goal = 0.2 * (self.scenario.goals[0] - self.scenario.formation_center) + \
                        self.scenario.formation_center
        self.env.rew_coeff = self.rew_coeff
        self.env.update_env(*self.room_dims)  # update the room box

        obs = self.env.reset()
        # the relative spherical coordinates of the gates
        self.preceding_goals = self.scenario.gate_poses_spherical
        # obs = self.add_preceding_gates_obs(obs_drone)

        # the initial take-off stage
        self.env.controller = NonlinearPositionController(self.env.dynamics, tf_control=False)
        for _ in range(20):
            self.env.step(np.zeros(4))
        self.env.goal = self.scenario.gate_poses[self.scenario.curr_gate_ptr].position.to_numpy_array()
        for _ in range(30):
            self.env.step(np.zeros(4))
        self.env.controller = OmegaThrustControl(self.env.dynamics)

        if self.use_obstacles:
            self.set_obstacles = np.zeros(self.obstacle_num, dtype=bool)
            quads_pos = np.array([self.env.dynamics.pos])
            quads_vel = np.array([self.env.dynamics.vel])
            # reset the racing track with a re-generated set of gates
            obs = self.multi_obstacles.reset(obs=obs, quads_pos=quads_pos, quads_vel=quads_vel,
                                             set_obstacles=self.set_obstacles, formation_size=self.quads_formation_size,
                                             goal_central=self.goal_central, )
            self.obst_quad_collisions_per_episode = 0
            self.prev_obst_quad_collisions = []

        self.all_collisions = {val: [0.0] for val in ['drone', 'ground', 'gates']}
        self.collisions_per_episode = self.collisions_after_settle = 0
        self.prev_drone_collisions, self.curr_drone_collisions = [], []

        self.reset_scene = True
        self.crashes_last_episode = 0

        return obs

    # noinspection PyTypeChecker
    def step(self, action):
        self.env.rew_coeff = self.rew_coeff
        obs, reward, done, info = self.env.step(action)
        self.drone_pos = drone_pos = self.env.dynamics.pos
        # run the scenario passed to self.quads_mode, we calculate the drone's racing state \
        # including collision, out-of-sight and goal-reached
        infos, rewards, racing_done = self.scenario.step(info, reward, drone_pos)
        # obs with the updated drone and gates' dynamic states
        # obs = self.add_preceding_gates_obs(obs_drone)
        # reward shaping for drone racing
        r_progress = infos['rewards']['rewraw_precede'] * self.rew_coeff['progress']
        r_safety = infos['rewards']['rewraw_safety'] * self.rew_coeff['safety']
        # r_pos = infos['rewards']['rewraw_pos'] * self.rew_coeff['pos']
        r_spin = infos['rewards']['rewraw_spin'] * self.rew_coeff['spin']
        r_effort = infos['rewards']['rewraw_action'] * self.rew_coeff['effort']
        r_orient = infos['rewards']['rewraw_orient'] * self.rew_coeff['orient']
        r_crash = infos['rewards']['rewraw_crash'] * self.rew_coeff['crash']
        r_terminal = infos['rewards']['rewraw_terminal']
        # mask the shifted dist progress when switching the tracking target
        if self.scenario.curr_gate_ptr != self.scenario.prev_gate_ptr:
            curr_tracking_gate = self.scenario.gate_poses[self.scenario.curr_gate_ptr]
            relative_pos = curr_tracking_gate.position.to_numpy_array() - self.drone_pos
            self.env.dist_prev = np.linalg.norm(relative_pos)
            self.scenario.prev_gate_ptr = self.scenario.curr_gate_ptr

        rewards = (r_progress + r_safety + r_spin + r_terminal + r_effort + r_orient + r_crash) * self.env.dt
        # print(rewards, r_progress, r_safety, r_spin, r_terminal, r_effort, r_orient, r_crash)

        self.env.goal = self.scenario.gate_poses[self.scenario.curr_gate_ptr].position.to_numpy_array()
        # Collisions with ground
        ground_collisions = [1.0 if pos[2] < 0.25 else 0.0 for pos in (self.drone_pos,)]
        self.all_collisions = {'ground': ground_collisions, }

        if done or racing_done:
            env_done = True
        else:
            env_done = False

        return obs, rewards, env_done, infos

    def render(self, mode='human', verbose=False):
        models = tuple(e.dynamics.model for e in (self.env,))

        if self.scene is None:
            self.init_scene_multi()
        # reset the rendered scene after the env is reset
        if self.reset_scene:
            self.scene.update_models(models)  # update the dynamics model
            self.scene.formation_size = self.quads_formation_size
            self.scene.update_env(self.room_dims)  # update/create all the assets of the scene
            self.scene.reset(self.scenario.gate_poses, self.all_dynamics(), self.multi_obstacles, self.all_collisions)

            self.reset_scene = False

        if self.quads_mode == "mix":
            self.scene.formation_size = self.scenario.scenario.formation_size
        else:
            self.scene.formation_size = self.scenario.formation_size
        self.frames_since_last_render += 1

        if self.render_skip_frames > 0:
            self.render_skip_frames -= 1
            return None

        # this is to handle the 1st step of the simulation that will typically be very slow
        if self.simulation_start_time > 0:
            simulation_time = time.time() - self.simulation_start_time
        else:
            simulation_time = 0

        realtime_control_period = 1 / self.control_freq

        render_start = time.time()
        frame = self.scene.render_chase(all_dynamics=self.all_dynamics(), goals=self.scenario.gate_poses,
                                        collisions=self.all_collisions,
                                        mode=mode, multi_obstacles=self.multi_obstacles)
        # Update the formation size of the scenario
        if self.quads_mode == "mix":
            self.scenario.scenario.update_formation_size(self.scene.formation_size)
        else:
            self.scenario.update_formation_size(self.scene.formation_size)

        render_time = time.time() - render_start

        desired_time_between_frames = realtime_control_period * self.frames_since_last_render / self.render_speed
        time_to_sleep = desired_time_between_frames - simulation_time - render_time

        # wait so we don't simulate/render faster than realtime
        if mode == "human" and time_to_sleep > 0:
            time.sleep(time_to_sleep)

        if simulation_time + render_time > desired_time_between_frames:
            self.render_every_nth_frame += 1
            if verbose:
                print(f"Last render + simulation time {render_time + simulation_time:.3f}")
                print(f"Rendering does not keep up, rendering every {self.render_every_nth_frame} frames")
        elif simulation_time + render_time < realtime_control_period * (
                self.frames_since_last_render - 1) / self.render_speed:
            self.render_every_nth_frame -= 1
            if verbose:
                print(f"We can increase rendering framerate, rendering every {self.render_every_nth_frame} frames")

        if self.render_every_nth_frame > 5:
            self.render_every_nth_frame = 5
            if self.env.tick % 20 == 0 and verbose:
                print(f"Rendering cannot keep up! Rendering every {self.render_every_nth_frame} frames")

        self.render_skip_frames = self.render_every_nth_frame - 1
        self.frames_since_last_render = 0

        self.simulation_start_time = time.time()

        if mode == "rgb_array":
            return frame

    def __deepcopy__(self, memo):
        """OpenGL scene can't be copied naively."""

        cls = self.__class__
        copied_env = cls.__new__(cls)
        memo[id(self)] = copied_env

        # this will actually break the reward shaping functionality in PBT, but we need to fix it in SampleFactory, not here
        skip_copying = {"scene", "reward_shaping_interface"}

        for k, v in self.__dict__.items():
            if k not in skip_copying:
                setattr(copied_env, k, deepcopy(v, memo))

        # warning! deep-copied env has its scene uninitialized! We gotta reuse one from the existing env
        # to avoid creating tons of windows
        copied_env.scene = None

        return copied_env
