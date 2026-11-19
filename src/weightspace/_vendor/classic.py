"""Classic-control dynamics and expert policies from WeightBench.

Source: https://github.com/Aayush-Marasini/WeightBench/tree/2acd60172fc9758eb9030b1cab4a2ff29be5c993/weightbench/tracks/tr02_control_lib
Copyright (c) 2026 WeightBench contributors. See WEIGHTBENCH_LICENSE.
The dynamics follow Gymnasium 1.x under the MIT license. Start bounds,
expert policies and task success criteria come from WeightBench.
"""
from __future__ import annotations

from typing import Callable

import numpy as np


class ClassicEnv:
    name: str
    state_names: list[str]
    obs_names: list[str]
    discrete: bool
    n_out: int
    horizon: int
    start_low: np.ndarray
    start_high: np.ndarray

    def sample_starts(self, rng: np.random.Generator, n: int) -> np.ndarray:
        return rng.uniform(self.start_low, self.start_high, size=(n, len(self.start_low)))

    def rollout(
        self,
        predict: Callable[[np.ndarray], np.ndarray],
        starts: np.ndarray,
        *,
        record: bool = False,
    ) -> dict:
        """Evaluate one policy row for each episode still running at a step."""
        states = np.array(starts, np.float64, copy=True)
        count = len(states)
        if record and count != 1:
            raise ValueError("trajectory recording requires one start")
        active = np.arange(count)
        returns = np.zeros(count)
        steps = np.full(count, self.horizon)
        terminated = np.zeros(count, bool)
        extra = self.init_info(count)
        trajectory = {"obs": [], "action": [], "reward": []}
        for t in range(self.horizon):
            obs = self.obs(states[active])
            actions = self.action(predict(self.norm(obs)))
            next_states, rewards, done = self.step(states[active], actions)
            if record:
                trajectory["obs"].append(obs[0])
                trajectory["action"].append(actions[0])
                trajectory["reward"].append(rewards[0])
            states[active] = next_states
            returns[active] += rewards
            self.update_info(extra, next_states, active)
            ended = active[done]
            steps[ended] = t + 1
            terminated[ended] = True
            active = active[~done]
            if not len(active):
                break
        info = {"ret": returns, "t_end": steps, "terminated": terminated, **extra}
        info["success"] = self.success(info)
        if record:
            info.update({key: np.asarray(value) for key, value in trajectory.items()})
        return info

    def init_info(self, B: int) -> dict:
        return {}

    def update_info(self, info: dict, next_states: np.ndarray, active: np.ndarray) -> None:
        pass


class MountainCar(ClassicEnv):
    """Gymnasium MountainCar-v0 with position and velocity observations.

    The actions 0, 1 and 2 push left, coast and push right, respectively.
    """
    name = "MountainCar-v0"
    state_names = obs_names = ["position", "velocity"]
    discrete = True
    n_out = 3
    horizon = 200
    min_pos, max_pos, max_speed, goal_pos = -1.2, 0.6, 0.07, 0.5
    force, gravity = 0.001, 0.0025
    start_low = np.array([-0.9, -0.02])
    start_high = np.array([0.25, 0.02])
    X0 = -0.45   # expert threshold: "right of X0 -> always push right"

    def obs(self, s):
        return s.copy()

    def norm(self, o):
        return np.stack([(o[:, 0] + 0.3) / 0.9, o[:, 1] / 0.07], 1)

    def action(self, out):
        return np.argmax(out, axis=1)

    def step(self, s, a):
        x, v = s[:, 0], s[:, 1]
        v = v + (a - 1) * self.force + np.cos(3 * x) * (-self.gravity)
        v = np.clip(v, -self.max_speed, self.max_speed)
        x = x + v
        x = np.clip(x, self.min_pos, self.max_pos)
        v = np.where((x == self.min_pos) & (v < 0), 0.0, v)
        done = (x >= self.goal_pos) & (v >= 0)
        return np.stack([x, v], 1), -np.ones(len(x)), done

    def expert_target(self, o):
        """Flawed expert: push right if moving right OR already right of X0, else push left."""
        right = (o[:, 1] > 0) | (o[:, 0] > self.X0)
        return np.where(right, 2, 0)

    def success(self, info):
        return info["terminated"].copy()


def angle_normalize(x):
    return ((x + np.pi) % (2 * np.pi)) - np.pi


class Pendulum(ClassicEnv):
    """Gymnasium Pendulum-v1. state = [theta, theta_dot] (theta=0 upright), obs = [cos, sin, theta_dot]."""
    name = "Pendulum-v1"
    state_names = ["theta", "theta_dot"]
    obs_names = ["cos(theta)", "sin(theta)", "theta_dot"]
    discrete = False
    n_out = 1
    horizon = 200
    max_speed, max_torque, dt, g, m, l = 8.0, 2.0, 0.05, 10.0, 1.0, 1.0
    start_low = np.array([-np.pi, -1.0])
    start_high = np.array([np.pi, 1.0])
    K1, K2 = 6.0, 1.5        # expert: saturated PD towards upright
    hold_steps = 50          # success: |angle| < 0.3 rad for each of the last hold_steps steps

    def obs(self, s):
        return np.stack([np.cos(s[:, 0]), np.sin(s[:, 0]), s[:, 1]], 1)

    def norm(self, o):
        return np.stack([o[:, 0], o[:, 1], o[:, 2] / 8.0], 1)

    def action(self, out):
        return np.clip(out[:, 0], -self.max_torque, self.max_torque)

    def step(self, s, u):
        th, thdot = s[:, 0], s[:, 1]
        u = np.clip(u, -self.max_torque, self.max_torque)
        cost = angle_normalize(th) ** 2 + 0.1 * thdot ** 2 + 0.001 * u ** 2
        newthdot = thdot + (3 * self.g / (2 * self.l) * np.sin(th) + 3.0 / (self.m * self.l ** 2) * u) * self.dt
        newthdot = np.clip(newthdot, -self.max_speed, self.max_speed)
        newth = th + newthdot * self.dt
        return np.stack([newth, newthdot], 1), -cost, np.zeros(len(th), bool)

    def expert_target(self, o):
        th = np.arctan2(o[:, 1], o[:, 0])
        return np.clip(-self.K1 * th - self.K2 * o[:, 2], -2.0, 2.0)[:, None]

    def init_info(self, B):
        return {"upright_run": np.zeros(B, int)}

    def update_info(self, info, next_states, active):
        up = np.abs(angle_normalize(next_states[:, 0])) < 0.3
        info["upright_run"][active] = np.where(up, info["upright_run"][active] + 1, 0)

    def success(self, info):
        return info["upright_run"] >= self.hold_steps


class CartPole(ClassicEnv):
    """Gymnasium CartPole-v1 dynamics with Euler integration.

    State and observation both contain cart position and velocity, pole angle
    and angular velocity. The actions 0 and 1 push left and right.
    """
    name = "CartPole-v1"
    state_names = obs_names = ["x", "x_dot", "theta", "theta_dot"]
    discrete = True
    n_out = 2
    horizon = 500
    gravity, masscart, masspole, length, force_mag, tau = 9.8, 1.0, 0.1, 0.5, 10.0, 0.02
    theta_threshold = 12 * 2 * np.pi / 360
    x_threshold = 2.4
    start_low = np.array([-1.8, -1.2, -0.05, -0.05])
    start_high = np.array([1.8, 1.2, 0.05, 0.05])
    KV, KT, KW = 0.12, 1.0, 0.35   # expert: push right iff KV*x_dot + KT*theta + KW*theta_dot > 0 (no position term)

    def obs(self, s):
        return s.copy()

    def norm(self, o):
        return o / np.array([2.4, 2.0, 0.21, 2.0])

    def action(self, out):
        return np.argmax(out, axis=1)

    def step(self, s, a):
        x, x_dot, theta, theta_dot = s.T
        total_mass = self.masspole + self.masscart
        polemass_length = self.masspole * self.length
        force = np.where(a == 1, self.force_mag, -self.force_mag)
        costheta, sintheta = np.cos(theta), np.sin(theta)
        temp = (force + polemass_length * theta_dot ** 2 * sintheta) / total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.length * (4.0 / 3.0 - self.masspole * costheta ** 2 / total_mass))
        xacc = temp - polemass_length * thetaacc * costheta / total_mass
        x = x + self.tau * x_dot
        x_dot = x_dot + self.tau * xacc
        theta = theta + self.tau * theta_dot
        theta_dot = theta_dot + self.tau * thetaacc
        s2 = np.stack([x, x_dot, theta, theta_dot], 1)
        done = (x < -self.x_threshold) | (x > self.x_threshold) | (theta < -self.theta_threshold) | (
            theta > self.theta_threshold)
        # gymnasium: reward 1 for every step, including the terminating one
        return s2, np.ones(len(x)), done

    def expert_target(self, o):
        return (self.KV * o[:, 1] + self.KT * o[:, 2] + self.KW * o[:, 3] > 0).astype(int)

    def success(self, info):
        return ~info["terminated"]
