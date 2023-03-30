import numpy as np
import bezier
import copy

from gym_art.quadrotor_multi.quad_scenarios_utils import QUADS_PARAMS_DICT, update_formation_and_max_agent_per_layer, \
    update_layer_dist, get_formation_range, get_goal_by_formation, get_z_value, QUADS_MODE_LIST, \
    QUADS_MODE_LIST_OBSTACLES, QUADS_MODE_LIST_SIMPLE
from gym_art.quadrotor_multi.quad_utils import generate_points, get_grid_dim_number
from gym_art.quadrotor_multi.trajectory_utils import *

def create_scenario(quads_mode, envs, num_agents, room_dims, room_dims_callback, rew_coeff, quads_formation,
                    quads_formation_size, track_gate_nums=None):
    cls = eval('Scenario_' + quads_mode)
    scenario = cls(quads_mode, envs, num_agents, room_dims, room_dims_callback, rew_coeff, quads_formation,
                   quads_formation_size, track_gate_nums=track_gate_nums)
    return scenario


class QuadrotorScenario:
    def __init__(self, quads_mode, envs, num_agents, room_dims, room_dims_callback, rew_coeff, quads_formation,
                 quads_formation_size, track_gate_nums):
        self.quads_mode = quads_mode
        self.envs = envs
        self.num_agents = num_agents
        self.room_dims = room_dims
        self.set_room_dims = room_dims_callback  # usage example: self.set_room_dims((10, 10, 10))
        self.rew_coeff = rew_coeff
        self.goals = None

        #  Set formation, num_agents_per_layer, lowest_formation_size, highest_formation_size, formation_size,
        #  layer_dist, formation_center
        #  Note: num_agents_per_layer for scalability, the maximum number of agent per layer
        self.formation = quads_formation
        self.num_agents_per_layer = 8
        quad_arm = self.envs[0].dynamics.arm
        self.lowest_formation_size, self.highest_formation_size = 8 * quad_arm, 16 * quad_arm
        self.formation_size = quads_formation_size
        self.layer_dist = self.lowest_formation_size
        self.formation_center = np.array([0.0, 0.0, 2.0])

        # Reset episode time
        if self.quads_mode != 'mix':
            ep_time = QUADS_PARAMS_DICT[quads_mode][2]
        else:
            ep_time = QUADS_PARAMS_DICT['swap_goals'][2]

        # for env in self.envs:
        #     env.reset_ep_len(ep_time=ep_time)

        # Aux variables for scenario: circular configuration
        self.settle_count = np.zeros(self.num_agents)
        self.metric_of_settle = 2.0 * quad_arm
        # Aux variables for scenario: pursuit evasion
        self.interp = None

    def name(self):
        """
        :return: scenario name
        """
        return self.__class__.__name__

    def generate_goals(self, num_agents, formation_center=None, layer_dist=0.0):
        if formation_center is None:
            formation_center = np.array([0., 0., 2.])

        if self.formation.startswith("circle"):
            if num_agents <= self.num_agents_per_layer:
                real_num_per_layer = [num_agents]
            else:
                whole_layer_num = num_agents // self.num_agents_per_layer
                real_num_per_layer = [self.num_agents_per_layer for _ in range(whole_layer_num)]
                rest_num = num_agents % self.num_agents_per_layer
                if rest_num > 0:
                    real_num_per_layer.append(rest_num)

            pi = np.pi
            goals = []
            for i in range(num_agents):
                cur_layer_num_agents = real_num_per_layer[i // self.num_agents_per_layer]
                degree = 2 * pi * (i % cur_layer_num_agents) / cur_layer_num_agents
                pos_0 = self.formation_size * np.cos(degree)
                pos_1 = self.formation_size * np.sin(degree)
                goal = get_goal_by_formation(formation=self.formation, pos_0=pos_0, pos_1=pos_1, layer_pos=(i//self.num_agents_per_layer) * layer_dist)
                goals.append(goal)

            goals = np.array(goals)
            goals += formation_center
        elif self.formation == "sphere":
            goals = self.formation_size * np.array(generate_points(num_agents)) + formation_center
        elif self.formation.startswith("grid"):
            if num_agents <= self.num_agents_per_layer:
                dim_1, dim_2 = get_grid_dim_number(num_agents)
                dim_size_each_layer = [[dim_1, dim_2]]
            else:
                # whole layer
                whole_layer_num = num_agents // self.num_agents_per_layer
                max_dim_1, max_dim_2 = get_grid_dim_number(self.num_agents_per_layer)
                dim_size_each_layer = [[max_dim_1, max_dim_2] for _ in range(whole_layer_num)]

                # deal with the rest of the drones
                rest_num = num_agents % self.num_agents_per_layer
                if rest_num > 0:
                    dim_1, dim_2 = get_grid_dim_number(rest_num)
                    dim_size_each_layer.append([dim_1, dim_2])

            goals = []
            for i in range(num_agents):
                dim_1, dim_2 = dim_size_each_layer[i//self.num_agents_per_layer]
                pos_0 = self.formation_size * (i % dim_2)
                pos_1 = self.formation_size * (int(i / dim_2) % dim_1)
                goal = get_goal_by_formation(formation=self.formation, pos_0=pos_0, pos_1=pos_1, layer_pos=(i//self.num_agents_per_layer) * layer_dist)
                goals.append(goal)

            mean_pos = np.mean(goals, axis=0)
            goals = goals - mean_pos + formation_center
        elif self.formation.startswith("cube"):
            dim_size = np.power(num_agents, 1.0 / 3)
            floor_dim_size = int(dim_size)
            goals = []
            for i in range(num_agents):
                pos_0 = self.formation_size * (int(i / floor_dim_size) % floor_dim_size)
                pos_1 = self.formation_size * (i % floor_dim_size)
                goal = np.array([formation_center[2] + self.formation_size * (i // np.square(floor_dim_size)), pos_0, pos_1])
                goals.append(goal)

            mean_pos = np.mean(goals, axis=0)
            goals = goals - mean_pos + formation_center
        else:
            raise NotImplementedError("Unknown formation")

        return goals

    def update_formation_size(self, new_formation_size):
        if new_formation_size != self.formation_size:
            self.formation_size = new_formation_size if new_formation_size > 0.0 else 0.0
            self.goals = self.generate_goals(num_agents=self.num_agents, formation_center=self.formation_center, layer_dist=self.layer_dist)
            for i, env in enumerate(self.envs):
                env.goal = self.goals[i]

    def update_formation_and_relate_param(self):
        # Reset formation, num_agents_per_layer, lowest_formation_size, highest_formation_size, formation_size, layer_dist
        self.formation, self.num_agents_per_layer = update_formation_and_max_agent_per_layer(mode=self.quads_mode)
        # QUADS_PARAMS_DICT:
        # Key: quads_mode; Value: 0. formation, 1: [formation_low_size, formation_high_size], 2: episode_time
        lowest_dist, highest_dist = QUADS_PARAMS_DICT[self.quads_mode][1]
        self.lowest_formation_size, self.highest_formation_size = \
            get_formation_range(mode=self.quads_mode, formation=self.formation, num_agents=self.num_agents,
                                low=lowest_dist, high=highest_dist, num_agents_per_layer=self.num_agents_per_layer)

        self.formation_size = np.random.uniform(low=self.lowest_formation_size, high=self.highest_formation_size)
        self.layer_dist = update_layer_dist(low=self.lowest_formation_size, high=self.highest_formation_size)

    def step(self, infos, rewards, pos):
        raise NotImplementedError("Implemented in a specific scenario")

    def reset(self):
        # Reset formation and related parameters
        self.update_formation_and_relate_param()

        # Reset formation center
        self.formation_center = np.array([0.0, 0.0, 2.0])

        # Regenerate goals, we don't have to assign goals to the envs,
        # the reset function in quadrotor_multi.py would do that
        self.goals = self.generate_goals(num_agents=self.num_agents, formation_center=self.formation_center, layer_dist=self.layer_dist)
        np.random.shuffle(self.goals)

    def standard_reset(self):
        # Reset formation and related parameters
        self.update_formation_and_relate_param()

        # Reset formation center
        self.formation_center = np.array([0.0, 0.0, 2.0])

        # Regenerate goals, we don't have to assign goals to the envs,
        # the reset function in quadrotor_multi.py would do that
        self.goals = self.generate_goals(num_agents=self.num_agents, formation_center=self.formation_center, layer_dist=self.layer_dist)
        np.random.shuffle(self.goals)


class Scenario_static_same_goal(QuadrotorScenario):
    def update_formation_size(self, new_formation_size):
        pass

    def step(self, infos, rewards, pos):
        return infos, rewards


class Scenario_static_diff_goal(QuadrotorScenario):
    def step(self, infos, rewards, pos):
        return infos, rewards


class Scenario_dynamic_same_goal(QuadrotorScenario):
    def __init__(self, quads_mode, envs, num_agents, room_dims, room_dims_callback, rew_coeff, quads_formation,
                 quads_formation_size, track_gate_nums):
        super().__init__(quads_mode, envs, num_agents, room_dims, room_dims_callback, rew_coeff, quads_formation,
                         quads_formation_size, track_gate_nums)
        # teleport every [4.0, 6.0] secs
        duration_time = 5.0
        self.control_step_for_sec = int(duration_time * self.envs[0].control_freq)

    def update_formation_size(self, new_formation_size):
        pass

    def step(self, infos, rewards, pos):
        tick = self.envs[0].tick
        if tick % self.control_step_for_sec == 0 and tick > 0:
            box_size = self.envs[0].box
            x, y = np.random.uniform(low=-box_size, high=box_size, size=(2,))
            z = np.random.uniform(low=-0.5 * box_size, high=0.5 * box_size) + 2.0
            z = max(0.25, z)
            self.formation_center = np.array([x, y, z])
            self.goals = self.generate_goals(num_agents=self.num_agents, formation_center=self.formation_center, layer_dist=0.0)
            for i, env in enumerate(self.envs):
                env.goal = self.goals[i]

        return infos, rewards

    def reset(self):
        # Update duration time
        duration_time = np.random.uniform(low=8.0, high=8.0)
        self.control_step_for_sec = int(duration_time * self.envs[0].control_freq)

        # Reset formation, and parameters related to the formation; formation center; goals
        self.standard_reset()


class Scenario_dynamic_diff_goal(QuadrotorScenario):
    def __init__(self, quads_mode, envs, num_agents, room_dims, room_dims_callback, rew_coeff, quads_formation,
                 quads_formation_size, track_gate_nums):
        super().__init__(quads_mode, envs, num_agents, room_dims, room_dims_callback, rew_coeff, quads_formation,
                         quads_formation_size, track_gate_nums)
        # teleport every [4.0, 6.0] secs
        duration_time = 5.0
        self.control_step_for_sec = int(duration_time * self.envs[0].control_freq)

    def update_goals(self):
        # Reset formation and related parameters
        self.update_formation_and_relate_param()

        # Reset goals
        self.goals = self.generate_goals(num_agents=self.num_agents, formation_center=self.formation_center, layer_dist=self.layer_dist)
        np.random.shuffle(self.goals)

    def step(self, infos, rewards, pos):
        tick = self.envs[0].tick
        if tick % self.control_step_for_sec == 0 and tick > 0:
            box_size = self.envs[0].box
            x, y = np.random.uniform(low=-box_size, high=box_size, size=(2,))

            # Get z value, and make sure all goals will above the ground
            z = get_z_value(num_agents=self.num_agents, num_agents_per_layer=self.num_agents_per_layer,
                            box_size=box_size, formation=self.formation, formation_size=self.formation_size)

            self.formation_center = np.array([x, y, z])
            self.update_goals()

            # Update goals to envs
            for i, env in enumerate(self.envs):
                env.goal = self.goals[i]

        return infos, rewards

    def reset(self):
        # Update duration time
        duration_time = np.random.uniform(low=4.0, high=6.0)
        self.control_step_for_sec = int(duration_time * self.envs[0].control_freq)

        # Reset formation, and parameters related to the formation; formation center; goals
        self.standard_reset()


class Scenario_ep_lissajous3D(QuadrotorScenario):
    # Based on https://mathcurve.com/courbes3d.gb/lissajous3d/lissajous3d.shtml
    @staticmethod
    def lissajous3D(tick, a=0.03, b=0.01, c=0.01, n=2, m=2, phi=90, psi=90):
        x = a * np.sin(tick)
        y = b * np.sin(n * tick + phi)
        z = c * np.cos(m * tick + psi)
        return x, y, z

    def step(self, infos, rewards, pos):
        control_freq = self.envs[0].control_freq
        tick = self.envs[0].tick / control_freq
        x, y, z = self.lissajous3D(tick)
        goal_x, goal_y, goal_z = self.goals[0]
        x_new, y_new, z_new = x + goal_x, y + goal_y, z + goal_z
        self.goals = np.array([[x_new, y_new, z_new] for _ in range(self.num_agents)])

        for i, env in enumerate(self.envs):
            env.goal = self.goals[i]

        return infos, rewards

    def update_formation_size(self, new_formation_size):
        pass

    def reset(self):
        # Reset formation and related parameters
        self.update_formation_and_relate_param()

        # Generate goals
        self.formation_center = np.array([-2.0, 0.0, 2.0])  # prevent drones from crashing into the wall
        self.goals = self.generate_goals(num_agents=self.num_agents, formation_center=self.formation_center, layer_dist=0.0)


class Scenario_ep_rand_bezier(QuadrotorScenario):
    def step(self, infos, rewards, pos):
        # randomly sample new goal pos in free space and have the goal move there following a bezier curve
        tick = self.envs[0].tick
        control_freq = self.envs[0].control_freq
        num_secs = 5
        control_steps = int(num_secs * control_freq)
        t = tick % control_steps
        room_dims = np.array(self.room_dims) - self.formation_size
        # min and max distance the goal can spawn away from its current location. 30 = empirical upper bound on
        # velocity that the drones can handle.
        max_dist = min(4.0, max(room_dims))  # default is 30
        min_dist = max_dist / 2
        if tick % control_steps == 0 or tick == 1:
            # sample a new goal pos that's within the room boundaries and satisfies the distance constraint
            new_goal_found = False
            while not new_goal_found:
                low, high = np.array([-room_dims[0] / 2, -room_dims[1] / 2, 0]), np.array(
                    [room_dims[0] / 2, room_dims[1] / 2, room_dims[2]])
                # need an intermediate point for a deg=2 curve
                new_pos = np.random.uniform(low=-high, high=high, size=(2, 3)).reshape(3, 2)
                # add some velocity randomization = random magnitude * unit direction
                new_pos = new_pos * np.random.randint(min_dist, max_dist + 1) / np.linalg.norm(new_pos, axis=0)
                new_pos = self.goals[0].reshape(3, 1) + new_pos
                lower_bound = np.expand_dims(low, axis=1)
                upper_bound = np.expand_dims(high, axis=1)
                new_goal_found = (new_pos > lower_bound + 0.5).all() and (
                        new_pos < upper_bound - 0.5).all()  # check bounds that are slightly smaller than the room dims
            nodes = np.concatenate((self.goals[0].reshape(3, 1), new_pos), axis=1)
            nodes = np.asfortranarray(nodes)
            pts = np.linspace(0, 1, control_steps)
            curve = bezier.Curve(nodes, degree=2)
            self.interp = curve.evaluate_multi(pts)
            # self.interp = np.clip(self.interp, a_min=np.array([0,0,0.2]).reshape(3,1), a_max=high.reshape(3,1)) # want goal clipping to be slightly above the floor
        if tick % control_steps != 0 and tick > 1:
            self.goals = np.array([self.interp[:, t] for _ in range(self.num_agents)])

            for i, env in enumerate(self.envs):
                env.goal = self.goals[i]

        return infos, rewards

    def update_formation_size(self, new_formation_size):
        pass


class Scenario_swap_goals(QuadrotorScenario):
    def __init__(self, quads_mode, envs, num_agents, room_dims, room_dims_callback, rew_coeff, quads_formation,
                 quads_formation_size, track_gate_nums):
        super().__init__(quads_mode, envs, num_agents, room_dims, room_dims_callback, rew_coeff, quads_formation,
                         quads_formation_size, track_gate_nums)
        # teleport every [4.0, 6.0] secs
        duration_time = 5.0
        self.control_step_for_sec = int(duration_time * self.envs[0].control_freq)

    def update_goals(self):
        np.random.shuffle(self.goals)
        for env, goal in zip(self.envs, self.goals):
            env.goal = goal

    def step(self, infos, rewards, pos):
        tick = self.envs[0].tick
        # Switch every [4, 6] seconds
        if tick % self.control_step_for_sec == 0 and tick > 0:
            self.update_goals()

        return infos, rewards

    def reset(self):
        # Update duration time
        duration_time = np.random.uniform(low=4.0, high=6.0)
        self.control_step_for_sec = int(duration_time * self.envs[0].control_freq)

        # Reset formation, and parameters related to the formation; formation center; goals
        self.standard_reset()


class Scenario_circular_config(QuadrotorScenario):
    def update_goals(self):
        np.random.shuffle(self.goals)
        for env, goal in zip(self.envs, self.goals):
            env.goal = goal

    def step(self, infos, rewards, pos):
        for i, e in enumerate(self.envs):
            dist = np.linalg.norm(pos[i] - e.goal)
            if abs(dist) < self.metric_of_settle:
                self.settle_count[i] += 1
            else:
                self.settle_count = np.zeros(self.num_agents)
                break

        # drones settled at the goal for 1 sec
        control_step_for_one_sec = int(self.envs[0].control_freq)
        tmp_count = self.settle_count >= control_step_for_one_sec
        if all(tmp_count):
            self.update_goals()
            rews_settle_raw = control_step_for_one_sec
            rews_settle = self.rew_coeff["quadsettle"] * rews_settle_raw
            for i, env in enumerate(self.envs):
                env.goal = self.goals[i]
                # Add settle rewards
                rewards[i] += rews_settle
                infos[i]["rewards"]["rew_quadsettle"] = rews_settle
                infos[i]["rewards"]["rewraw_quadsettle"] = rews_settle_raw

            self.settle_count = np.zeros(self.num_agents)

        return infos, rewards


class Scenario_dynamic_formations(QuadrotorScenario):
    def __init__(self, quads_mode, envs, num_agents, room_dims, room_dims_callback, rew_coeff, quads_formation,
                 quads_formation_size, track_gate_nums):
        super().__init__(quads_mode, envs, num_agents, room_dims, room_dims_callback, rew_coeff, quads_formation,
                         quads_formation_size, track_gate_nums)
        # if increase_formation_size is True, increase the formation size
        # else, decrease the formation size
        self.increase_formation_size = True
        # low: 0.1m/s, high: 0.3m/s
        self.control_speed = np.random.uniform(low=1.0, high=3.0)

    # change formation sizes on the fly
    def update_goals(self):
        self.goals = self.generate_goals(self.num_agents, self.formation_center, layer_dist=self.layer_dist)
        for env, goal in zip(self.envs, self.goals):
            env.goal = goal

    def step(self, infos, rewards, pos):
        if self.formation_size <= -self.highest_formation_size:
            self.increase_formation_size = True
            self.control_speed = np.random.uniform(low=1.0, high=3.0)
        elif self.formation_size >= self.highest_formation_size:
            self.increase_formation_size = False
            self.control_speed = np.random.uniform(low=1.0, high=3.0)

        if self.increase_formation_size:
            self.formation_size += 0.001 * self.control_speed
        else:
            self.formation_size -= 0.001 * self.control_speed

        self.update_goals()
        return infos, rewards

    def reset(self):
        self.increase_formation_size = True if np.random.uniform(low=0.0, high=1.0) < 0.5 else False
        self.control_speed = np.random.uniform(low=1.0, high=3.0)

        # Reset formation, and parameters related to the formation; formation center; goals
        self.standard_reset()

    def update_formation_size(self, new_formation_size):
        if new_formation_size != self.formation_size:
            self.formation_size = new_formation_size if new_formation_size > 0.0 else 0.0
            self.update_goals()


class Scenario_swarm_vs_swarm(QuadrotorScenario):
    def __init__(self, quads_mode, envs, num_agents, room_dims, room_dims_callback, rew_coeff, quads_formation,
                 quads_formation_size, track_gate_nums):
        super().__init__(quads_mode, envs, num_agents, room_dims, room_dims_callback, rew_coeff, quads_formation,
                         quads_formation_size, track_gate_nums)
        # teleport every [4.0, 6.0] secs
        duration_time = 5.0
        self.control_step_for_sec = int(duration_time * self.envs[0].control_freq)
        self.goals_1, self.goals_2 = None, None
        self.goal_center_1, self.goal_center_2 = None, None

    def formation_centers(self):
        if self.formation_center is None:
            self.formation_center = np.array([0., 0., 2.])

        # self.envs[0].box = 2.0
        box_size = self.envs[0].box
        dist_low_bound = self.lowest_formation_size
        # Get the 1st goal center
        x, y = np.random.uniform(low=-box_size, high=box_size, size=(2,))
        # Get z value, and make sure all goals will above the ground
        z = get_z_value(num_agents=self.num_agents, num_agents_per_layer=self.num_agents_per_layer,
                        box_size=box_size, formation=self.formation, formation_size=self.formation_size)

        goal_center_1 = np.array([x, y, z])

        # Get the 2nd goal center
        goal_center_distance = np.random.uniform(low=box_size/4, high=box_size)

        phi = np.random.uniform(low=-np.pi, high=np.pi)
        theta = np.random.uniform(low=-0.5 * np.pi, high=0.5 * np.pi)
        goal_center_2 = goal_center_1 + goal_center_distance * np.array(
            [np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)])
        diff_x, diff_y, diff_z = goal_center_2 - goal_center_1
        if self.formation.endswith("horizontal"):
            if abs(diff_z) < dist_low_bound:
                goal_center_2[2] = np.sign(diff_z) * dist_low_bound + goal_center_1[2]
        elif self.formation.endswith("vertical_xz"):
            if abs(diff_y) < dist_low_bound:
                goal_center_2[1] = np.sign(diff_y) * dist_low_bound + goal_center_1[1]
        elif self.formation.endswith("vertical_yz"):
            if abs(diff_x) < dist_low_bound:
                goal_center_2[0] = np.sign(diff_x) * dist_low_bound + goal_center_1[0]

        return goal_center_1, goal_center_2

    def create_formations(self, goal_center_1, goal_center_2):
        self.goals_1 = self.generate_goals(num_agents=self.num_agents // 2, formation_center=goal_center_1, layer_dist=self.layer_dist)
        self.goals_2 = self.generate_goals(num_agents=self.num_agents - self.num_agents // 2, formation_center=goal_center_2, layer_dist=self.layer_dist)
        self.goals = np.concatenate([self.goals_1, self.goals_2])

    def update_goals(self):
        tmp_goal_center_1 = copy.deepcopy(self.goal_center_1)
        tmp_goal_center_2 = copy.deepcopy(self.goal_center_2)
        self.goal_center_1 = tmp_goal_center_2
        self.goal_center_2 = tmp_goal_center_1

        self.update_formation_and_relate_param()
        self.create_formations(self.goal_center_1, self.goal_center_2)
        # Shuffle goals
        np.random.shuffle(self.goals_1)
        np.random.shuffle(self.goals_2)
        self.goals = np.concatenate([self.goals_1, self.goals_2])
        for i, env in enumerate(self.envs):
            env.goal = self.goals[i]

    def step(self, infos, rewards, pos):
        tick = self.envs[0].tick
        # Switch every [4, 6] seconds
        if tick % self.control_step_for_sec == 0 and tick > 0:
            self.update_goals()
        return infos, rewards

    def reset(self):
        # Update duration time
        duration_time = np.random.uniform(low=4.0, high=6.0)
        self.control_step_for_sec = int(duration_time * self.envs[0].control_freq)

        # Reset formation and related parameters
        self.update_formation_and_relate_param()

        # Reset the formation size and the goals of swarms
        self.goal_center_1, self.goal_center_2 = self.formation_centers()
        self.create_formations(self.goal_center_1, self.goal_center_2)

        # This is for initialize the pos for obstacles
        self.formation_center = (self.goal_center_1 + self.goal_center_2) / 2

    def update_formation_size(self, new_formation_size):
        if new_formation_size != self.formation_size:
            self.formation_size = new_formation_size if new_formation_size > 0.0 else 0.0
            self.create_formations(self.goal_center_1, self.goal_center_2)
            for i, env in enumerate(self.envs):
                env.goal = self.goals[i]


class Scenario_tunnel(QuadrotorScenario):
    def update_goals(self, formation_center):
        self.goals = self.generate_goals(num_agents=self.num_agents, formation_center=formation_center, layer_dist=self.layer_dist)
        for env, goal in zip(self.envs, self.goals):
            env.goal = goal

    def step(self, infos, rewards, pos):
        # hack to make drones and goals be on opposite sides of the tunnel
        t = self.envs[0].tick
        if t == 1:
            for env in self.envs:
                if abs(env.goal[0]) > abs(env.goal[1]):
                    env.goal[0] = -env.goal[0]
                else:
                    env.goal[1] = -env.goal[1]
        return infos, rewards

    def reset(self):
        # tunnel could be in the x or y direction
        p = np.random.uniform(0, 1)
        if p <= 0.5:
            self.update_room_dims((10, 2, 2))
            formation_center = np.array([-4, 0, 1])
        else:
            self.update_room_dims((2, 10, 2))
            formation_center = np.array([0, -4, 1])
        self.update_goals(formation_center)


class Scenario_run_away(QuadrotorScenario):
    def __init__(self, quads_mode, envs, num_agents, room_dims, room_dims_callback, rew_coeff, quads_formation,
                 quads_formation_size, track_gate_nums):
        super().__init__(quads_mode, envs, num_agents, room_dims, room_dims_callback, rew_coeff, quads_formation,
                         quads_formation_size, track_gate_nums)

    def update_goals(self):
        self.goals = self.generate_goals(self.num_agents, self.formation_center, layer_dist=self.layer_dist)
        for env, goal in zip(self.envs, self.goals):
            env.goal = goal

    def step(self, infos, rewards, pos):
        tick = self.envs[0].tick
        control_step_for_sec = int(1.0 * self.envs[0].control_freq)

        if tick % control_step_for_sec == 0 and tick > 0:
            g_index = np.random.randint(low=1, high=self.num_agents, size=2)
            self.goals[0] = self.goals[g_index[0]]
            self.goals[1] = self.goals[g_index[1]]
            self.envs[0].goal = self.goals[0]
            self.envs[1].goal = self.goals[1]

        return infos, rewards

    def reset(self):
        # Reset formation and related parameters
        self.update_formation_and_relate_param()
        # Reset formation center
        self.formation_center = np.array([0.0, 0.0, 2.0])

        # Regenerate goals, we don't have to assign goals to the envs,
        # the reset function in quadrotor_multi.py would do that
        self.goals = self.generate_goals(num_agents=self.num_agents, formation_center=self.formation_center, layer_dist=self.layer_dist)
        np.random.shuffle(self.goals)

    def update_formation_size(self, new_formation_size):
        if new_formation_size != self.formation_size:
            self.formation_size = new_formation_size if new_formation_size > 0.0 else 0.0
            self.update_goals()


class Scenario_mix(QuadrotorScenario):
    def __init__(self, quads_mode, envs, num_agents, room_dims, room_dims_callback, rew_coeff, quads_formation,
                 quads_formation_size, track_gate_nums):
        super().__init__(quads_mode, envs, num_agents, room_dims, room_dims_callback, rew_coeff, quads_formation,
                         quads_formation_size, track_gate_nums)
        self.room_dims_callback = room_dims_callback

        obst_mode = self.envs[0].obstacle_mode

        # Once change the parameter here, should also update QUADS_PARAMS_DICT to make sure it is same as run a single scenario
        # key: quads_mode
        # value: 0. formation, 1: [formation_low_size, formation_high_size], 2: episode_time
        if num_agents == 1:
            self.quads_mode_list = QUADS_MODE_LIST_SIMPLE

        # Check whether you use obstacles or not
        if obst_mode == 'no_obstacles':
            self.quads_mode_list = QUADS_MODE_LIST
        else:
            self.quads_mode_list = QUADS_MODE_LIST_OBSTACLES

        # actual scenario being used
        self.scenario = None

    def name(self):
        """
        :return: the name of the actual scenario used in this episode
        """
        return self.scenario.__class__.__name__

    def step(self, infos, rewards, pos):
        infos, rewards = self.scenario.step(infos=infos, rewards=rewards, pos=pos)
        # This is set for obstacle mode
        self.goals = self.scenario.goals
        self.formation_size = self.scenario.formation_size
        return infos, rewards

    def reset(self):
        mode_index = np.random.randint(low=0, high=len(self.quads_mode_list))
        mode = self.quads_mode_list[mode_index]

        # Init the scenario
        self.scenario = create_scenario(quads_mode=mode, envs=self.envs, num_agents=self.num_agents,
                                        room_dims=self.room_dims, room_dims_callback=self.room_dims_callback,
                                        rew_coeff=self.rew_coeff, quads_formation=self.formation,
                                        quads_formation_size=self.formation_size)

        self.scenario.reset()
        self.goals = self.scenario.goals
        self.formation_size = self.scenario.formation_size


class Scenario_traverse_gate(QuadrotorScenario):
    def __init__(self, quads_mode, envs, num_agents, room_dims, room_dims_callback, rew_coeff, quads_formation,
                 quads_formation_size, track_gate_nums):
        super().__init__(quads_mode, envs, num_agents, room_dims, room_dims_callback, rew_coeff, quads_formation,
                         quads_formation_size, track_gate_nums)
        self.track_gate_nums = track_gate_nums
        self.formation_center = np.array([0.0, 0.0, 5.0])
        self.gate_poses = None
        self.gate_poses_spherical = None
        self.scenario_tick = 0
        self.curr_gate_ptr = 0
        self.prev_gate_ptr = 0
        self.prev_preceded_seg_len = 0
        self.gate_width = 0.5
        self.d_max = 2.5
        self.gate_pass_threshold = 0.36

    def update_goals(self):
        pass

    def step(self, infos, rewards, drone_pos):
        self.scenario_tick += 1
        curr_tracking_gate = self.gate_poses[self.curr_gate_ptr]
        relative_pos = curr_tracking_gate.position.to_numpy_array() - drone_pos
        # calculate the included angle between the relative pos vector and gate's normal in \
        # the gate's body frame, we always set the gate face toward the X axis.
        gate_normal_body = np.array([0, 1, 0])
        point_angle = included_angle(relative_pos, gate_normal_body)
        point_angle_degree = point_angle * 180 / np.pi
        racing_done = False

        # # reward shaping
        # calculate the preceded incremental distance for r_p
        if self.curr_gate_ptr == 0:
            g_1 = self.formation_center
        else:
            g_1 = self.gate_poses[self.curr_gate_ptr - 1].position.to_numpy_array()
        g_2 = self.gate_poses[self.curr_gate_ptr].position.to_numpy_array()
        curr_preceded_seg_len = (drone_pos - g_1).dot(g_2 - g_1) / np.linalg.norm(g_2 - g_1)
        curr_preceded_seg_percent = curr_preceded_seg_len / np.linalg.norm(g_2 - g_1)
        r_p = curr_preceded_seg_len - self.prev_preceded_seg_len

        self.prev_preceded_seg_len = curr_preceded_seg_len
        # calculate the distance from drone to the gate's plane and normal
        d_plane_g2 = np.linalg.norm(relative_pos) * np.cos(point_angle)
        f = np.maximum(1 - (d_plane_g2 / self.d_max), 0.0)
        d_normal_g2 = np.linalg.norm(relative_pos) * np.sin(point_angle)
        v = np.maximum((1 - f) * (self.gate_width / 6.0), 0.05)
        r_s = -f**2 * (1 - np.exp(-d_normal_g2**2 / v * 0.5))

        infos['rewards']['rewraw_progress'] = r_p if self.scenario_tick > 1 else 0.0
        infos['rewards']['rewraw_safety'] = r_s

        # # check whether the drone is out of the sight of the gate
        # out_of_sight = False
        # projected_angle = included_angle(drone_pos - g_1, gate_normal_body)
        # d_plane_g1 = np.linalg.norm(drone_pos - g_1) * np.cos(projected_angle)
        # d_normal_g1 = np.linalg.norm(drone_pos - g_1) * np.sin(projected_angle)
        # if projected_angle > 90.0 / 2.0 or point_angle > 90.0 / 2.0:
        #     if d_plane_g1 < self.gate_width and d_normal_g1 < self.gate_width:
        #         print(np.degrees(projected_angle), np.degrees(point_angle), d_plane_g1, d_normal_g1)
        #         out_of_sight = False
        #     elif d_plane_g2 < self.gate_width and d_normal_g2 < self.gate_width:
        #         print(np.degrees(projected_angle), np.degrees(point_angle), d_plane_g2, d_normal_g2)
        #         out_of_sight = False
        #     else:
        #         out_of_sight = True
        #         print('the drone is out of sight of the gate when wandering')
        # if out_of_sight:
        #     racing_done = True
        #     return infos, rewards, racing_done

        # # check whether the drone is out of the sight of the gate
        relative_dist = np.linalg.norm(relative_pos)
        if relative_dist > R_RANGE[-1] + self.gate_width:
            racing_done = True
            r_terminal = -np.minimum((relative_dist / self.gate_width) ** 2, 20)
            infos['rewards']['rewraw_terminal'] = r_terminal
            print('the drone is out of the range of the gate {0} at {1} env step'.format(self.curr_gate_ptr, self.scenario_tick))
            return infos, rewards, racing_done

        # # check whether the drone has passed the gate, collided or just passed by.
        quad_arm = infos['dyn_params']['arm'][0]  # default value is 0.0459

        r_terminal = 0
        if curr_preceded_seg_percent > 0.96:
            if d_normal_g2 > self.gate_width / 2 + quad_arm:
                racing_done = True
                # r_terminal = -np.minimum((np.linalg.norm(relative_pos) / self.gate_width) ** 2, 20)
                infos['rewards']['rewraw_terminal'] = r_terminal
                print('the drone just miss the gate {0} at {1} env step'.format(self.curr_gate_ptr, self.scenario_tick))
                return infos, rewards, racing_done
            elif self.gate_width / 2 - quad_arm < d_normal_g2 <= self.gate_width / 2 + quad_arm:
                racing_done = True
                r_terminal = -np.minimum((np.linalg.norm(relative_pos) / self.gate_width) ** 2, 20)
                infos['rewards']['rewraw_terminal'] = r_terminal
                print('the drone has collided the gate {0} at {1} env step'.format(self.curr_gate_ptr, self.scenario_tick))
                return infos, rewards, racing_done
            else:
                self.prev_gate_ptr = self.curr_gate_ptr
                self.curr_gate_ptr += 1
                self.prev_preceded_seg_len = 0
                r_terminal = np.minimum(5 * (self.gate_width / d_normal_g2), 20)
                if self.curr_gate_ptr == self.track_gate_nums:
                    racing_done = True
                    infos['rewards']['rewraw_terminal'] = r_terminal
                    print('the drone has passed all the gates', self.scenario_tick)
                    return infos, rewards, racing_done
                else:
                    infos['rewards']['rewraw_terminal'] = r_terminal
                    print('the drone has passed the gate {0} at {1} env step.'.format(self.curr_gate_ptr, self.scenario_tick))
                    return infos, rewards, racing_done

        infos['rewards']['rewraw_terminal'] = r_terminal

        return infos, rewards, racing_done

    def reset(self):
        # Reset formation and related parameters
        self.update_formation_and_relate_param()
        self.curr_gate_ptr = 0
        self.prev_preceded_seg_len = 0
        self.scenario_tick = 0
        # generate a new random track
        # gate_poses: list of Vector3, gate_poses_spherical: list of (r, theta, phi)
        gate_poses, gate_poses_spherical = generate_random_track(self.formation_center, self.track_gate_nums)
        curr_tracking_goal = gate_poses[0].position
        self.goals = (curr_tracking_goal.to_numpy_array(), )
        self.gate_poses = gate_poses
        self.gate_poses_spherical = gate_poses_spherical

    def update_formation_size(self, new_formation_size):
        if new_formation_size != self.formation_size:
            self.formation_size = new_formation_size if new_formation_size > 0.0 else 0.0
            self.update_goals()
