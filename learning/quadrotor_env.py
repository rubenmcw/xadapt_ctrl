import gym
import numpy as np
from gym import spaces
from ruamel.yaml import YAML
from py3dmath import Vec3, Rotation

from uav_sim.vehicle import Vehicle
from uav_sim.positioncontroller import PositionController
from uav_sim.attitudecontroller import QuadcopterAttitudeControllerNested
from uav_sim.mixer import QuadcopterMixer


class QuadrotorEnv(gym.Env):
    """Simple quadrotor environment wrapping :class:`uav_sim` components.

    This environment is intentionally light weight.  It simulates a
    quadrotor using the provided vehicle dynamics and controllers.  The
    observation is a 34 dimensional vector as specified in
    ``learning/hyperparam.yaml``.  The semantics of many of those
    observation entries are placeholders for now; the goal of this class
    is simply to provide an interface that can be consumed by the PPO
    training code in :mod:`learning.train`.
    """

    metadata = {"render.modes": []}

    def __init__(self, cfg_yaml: str, render: bool = False):
        super().__init__()

        # Load configuration
        yaml = YAML().load(cfg_yaml) if isinstance(cfg_yaml, str) else cfg_yaml
        self.dt = yaml["simulation"].get("sim_dt", 0.002)
        self.obs_size = yaml["observation_space"].get("env_obs_size", 34)
        self.reward_cfg = yaml.get("rewards", {})

        # Spaces
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.obs_size,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(4,), dtype=np.float32
        )

        # Place holders for RMS normalisation (used by training code)
        self.obs_mean = np.zeros(self.obs_size, dtype=np.float64)
        self.obs_var = np.ones(self.obs_size, dtype=np.float64)

        self.max_motor_thrust = 10.0  # Newtons, arbitrary
        self.vehicle = None
        self.last_action = np.zeros(4, dtype=np.float32)
        self.seed()

    # ------------------------------------------------------------------
    # Helpers expected by ``learning/train.py``
    def seed(self, seed: int | None = None):
        self.np_random = np.random.default_rng(seed)
        return [seed]

    def load_rms(self, path: str):
        """Stub for loading running mean/variance stats."""
        try:
            data = np.load(path)
            self.obs_mean = data.get("mean", self.obs_mean)
            self.obs_var = data.get("var", self.obs_var)
        except Exception:
            # If anything goes wrong simply keep defaults.
            pass

    def get_obs_norm(self):
        return self.obs_mean, self.obs_var

    # ------------------------------------------------------------------
    # Gym API
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)

        # Instantiate a new vehicle each reset
        mass = 1.0
        inertia = np.diag([0.01, 0.01, 0.02])
        omega_sqr_to_drag = 1e-5
        dist_torque_std = 0.0
        self.vehicle = Vehicle(mass, inertia, omega_sqr_to_drag, dist_torque_std)

        # Add four motors in a cross configuration
        l = 0.1  # arm length in meters
        speed_sqr_to_thrust = 1e-5
        speed_sqr_to_torque = 1e-6
        time_const = 0.02
        inertia_motor = 1e-5
        positions = [
            Vec3(l, 0, 0),
            Vec3(0, l, 0),
            Vec3(-l, 0, 0),
            Vec3(0, -l, 0),
        ]
        spins = [Vec3(0, 0, 1), Vec3(0, 0, -1), Vec3(0, 0, 1), Vec3(0, 0, -1)]
        for pos, spin in zip(positions, spins):
            self.vehicle.add_motor(
                motorPosition=pos,
                spinDir=spin,
                minSpeed=0.0,
                maxSpeed=2000.0,
                speedSqrToThrust=speed_sqr_to_thrust,
                speedSqrToTorque=speed_sqr_to_torque,
                timeConst=time_const,
                inertia=inertia_motor,
            )

        # Randomise initial state
        self.vehicle.set_position(
            Vec3(*self.np_random.uniform(-0.1, 0.1, size=3))
        )
        self.vehicle.set_velocity(
            Vec3(*self.np_random.uniform(-0.1, 0.1, size=3))
        )
        rand_rot = Vec3(*self.np_random.uniform(-0.1, 0.1, size=3))
        self.vehicle.set_attitude(Rotation.from_rotation_vector(rand_rot))
        self.vehicle._omega = Vec3(*self.np_random.uniform(-0.1, 0.1, size=3))
        self.vehicle._accel = Vec3(0, 0, 0)

        self.last_action = np.zeros(4, dtype=np.float32)
        return self._get_obs(), {}

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        self.last_action = np.clip(action, -1.0, 1.0)
        motor_cmds = (self.last_action + 1.0) / 2.0 * self.max_motor_thrust

        # Run vehicle dynamics
        self.vehicle.run(self.dt, motor_cmds)

        obs = self._get_obs()

        # Reward components
        roll_pen = self.reward_cfg.get("roll_vel_coeff", 0.0) * (
            self.vehicle._omega.x ** 2
        )
        pitch_pen = self.reward_cfg.get("pitch_vel_coeff", 0.0) * (
            self.vehicle._omega.y ** 2
        )
        yaw_pen = self.reward_cfg.get("yaw_vel_coeff", 0.0) * (
            self.vehicle._omega.z ** 2
        )
        lin_acc = self.vehicle._accel
        lin_acc_pen = self.reward_cfg.get("lin_accel_coeff", 0.0) * (
            lin_acc.x**2 + lin_acc.y**2 + lin_acc.z**2
        )
        survive = self.reward_cfg.get("survive_coeff", 0.0)
        oscillate = 0.0  # Placeholder for oscillation term

        done = False
        crash_pen = 0.0
        if self.vehicle._pos.z < 0.0:
            done = True
            crash_pen = self.reward_cfg.get("crash_coeff", 0.0)

        reward = roll_pen + pitch_pen + yaw_pen + lin_acc_pen + survive + oscillate + crash_pen

        info = {
            "roll_vel_penalty": roll_pen,
            "pitch_vel_penalty": pitch_pen,
            "yaw_vel_penalty": yaw_pen,
            "lin_accel_penalty": lin_acc_pen,
            "survive_reward": survive,
            "oscillate_reward": oscillate,
            "total": reward,
        }

        return obs, reward, done, info

    # ------------------------------------------------------------------
    def _get_obs(self):
        """Construct the observation vector."""
        obs = np.zeros(self.obs_size, dtype=np.float32)

        pos = self.vehicle._pos
        vel = self.vehicle._vel
        rot_vec = self.vehicle._att.to_rotation_vector()
        omega = self.vehicle._omega
        accel = self.vehicle._accel
        motor_speeds = self.vehicle.get_motor_speeds()

        obs[0:3] = [pos.x, pos.y, pos.z]
        obs[3:6] = [vel.x, vel.y, vel.z]
        obs[6:9] = [rot_vec.x, rot_vec.y, rot_vec.z]
        obs[9:12] = [omega.x, omega.y, omega.z]
        obs[12:15] = [accel.x, accel.y, accel.z]
        obs[15:19] = motor_speeds
        # Remaining entries left as zero placeholders
        return obs