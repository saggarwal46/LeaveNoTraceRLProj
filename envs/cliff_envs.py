# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from gym.envs.mujoco.half_cheetah import HalfCheetahEnv
from gym.envs.mujoco.walker2d import Walker2dEnv
from gym.envs.mujoco import mujoco_env
from gym.wrappers import Monitor

import numpy as np
import os


def tolerance(x, bounds, margin):
    '''Returns 1 when x is within the bounds, and decays sigmoidally
    when x is within a certain margin outside the bounds.
    We've copied the function from [1] to reduce dependencies.

    [1] Tassa, Yuval, et al. "DeepMind Control Suite." arXiv preprint
    arXiv:1801.00690 (2018).
    '''
    (lower, upper) = bounds
    if lower <= x <= upper:
        return 0
    elif x < lower:
        dist_from_margin = lower - x
    else:
        assert x > upper
        dist_from_margin = x - upper
    loss_at_margin = 0.95
    w = np.arctanh(np.sqrt(loss_at_margin)) / margin
    s = np.tanh(w * dist_from_margin)
    return s*s


def huber(x, p):
    return np.sqrt(x*x + p*p) - p


class CliffCheetahEnv(HalfCheetahEnv):
    def __init__(self):
        envs_folder = os.path.dirname(os.path.abspath(__file__))
        xml_filename = os.path.join(envs_folder,
                                    'assets/cliff_cheetah.xml')
        mujoco_env.MujocoEnv.__init__(self, xml_filename, 5)

    def step(self, a):
        (s, _, done, info) = super(CliffCheetahEnv, self).step(a)
        r = self._get_rewards(s, a)[0]
        return (s, r, done, info)

    def _get_obs(self):
        '''Modified to include the x coordinate.'''
        return np.concatenate([
            self.model.data.qpos.flat,
            self.model.data.qvel.flat,
        ])

    def _get_rewards(self, s, a):
        (x, z, theta) = s[:3]
        xvel = s[9]
        # Reward the forward agent for running 9 - 11 m/s.
        forward_reward = (1.0 - tolerance(xvel, (9, 11), 7))
        theta_reward = 1.0 - tolerance(theta,
                                       bounds=(-0.05, 0.05),
                                       margin=0.1)
        # Reward the reset agent for being at the origin, plus
        # reward shaping to be near the origin and upright.
        reset_reward = 0.8 * (np.abs(x) < 0.5) + 0.1 * (1 - 0.2 * np.abs(x)) + 0.1 * theta_reward
        return (forward_reward, reset_reward)


class CliffWalkerEnv(Walker2dEnv):
    def __init__(self, with_reset_cost=False, use_custom_step=False):
        envs_folder = os.path.dirname(os.path.abspath(__file__))
        xml_filename = os.path.join(envs_folder,
                                    'assets/cliff_walker.xml')
        self.with_reset_cost = with_reset_cost
        self.use_custom_step = use_custom_step
        mujoco_env.MujocoEnv.__init__(self, xml_filename, 5)
    def step(self, a):
        (s, _, done, info) = super(CliffWalkerEnv, self).step(a)
        r = self._get_rewards(s, a)[0]
        return (s, r, done, info)

    def _get_obs(self):
        '''Modified to include the x coordinate.'''
        qpos = self.model.data.qpos
        qvel = self.model.data.qvel
        return np.concatenate([qpos[:], np.clip(qvel, -10, 10)]).ravel()
    
    def falling_off_cliff(self, s):
        x = s[0]
        torso_height = s[1]
        return torso_height < 0.7 and np.abs(x) > 5.8
    
    def fell_on_cliff(self, s):
        x = s[0]
        torso_height = s[1]
        return torso_height < 0.7 and np.abs(x) <= 5.8
    
    def reset_cost(self, s):
        if self.falling_off_cliff(s):
            return 1.0  # Off the cliff
        elif self.fell_on_cliff(s):
            return 0.3  # On the cliff but fallen
        else:
            return 0.0  # Minimal deviation

    def _get_rewards(self, s, a):
        x = s[0]
        running_vel = s[9] - 2.0
        torso_height = s[1]
        is_standing = float(torso_height > 1.2)
        is_falling = float(torso_height < 0.7)
        run_reward = np.clip(1 - 0.2 * huber(running_vel, p=0.1), 0, 1)
        stand_reward = np.clip(0.25 * torso_height +
                               0.25 * is_standing +
                               0.5 * (1 - is_falling), 0, 1)
        control_reward = np.clip(1 - 0.05 * np.dot(a, a), 0, 1)
        reset_location_reward = 0.8 * (np.abs(x) < 0.5) + 0.2 * (1 - 0.2 * np.abs(x))
        forward_reward = 0.5 * run_reward + 0.25 * stand_reward + 0.25 * control_reward
        reset_reward = 0.5 * reset_location_reward + 0.25 * stand_reward + 0.25 * control_reward
        
        if self.with_reset_cost:
            reset_cost = self.reset_cost(s)
            return forward_reward - reset_cost, reset_reward - reset_cost
        return (forward_reward, reset_reward)
    
    def _step(self, a):
        if not self.use_custom_step:
            return super(Walker2dEnv, self)._step(a)
        posbefore = self.model.data.qpos[0, 0]
        self.do_simulation(a, self.frame_skip)
        posafter, height, ang = self.model.data.qpos[0:3, 0]
        alive_bonus = 1.0
        reward = ((posafter - posbefore) / self.dt)
        reward += alive_bonus
        reward -= 1e-3 * np.square(a).sum()
        done = not (height < 2.0 and
                    ang > -1.0 and ang < 1.0)
        ob = self._get_obs()
        return ob, reward, done, {}


if __name__ == '__main__':
    import os
    import cv2
    import numpy as np
    from gym.envs.mujoco import Walker2dEnv
    from gym.wrappers import TimeLimit

    # Set the Mujoco headless rendering backend
    os.environ["MUJOCO_GL"] = "osmesa"  # Use 'osmesa' if 'egl' fails
    os.environ["DISPLAY"] = ""  # Unset DISPLAY to avoid GLFW attempting X11
    os.environ["LD_PRELOAD"] = "/usr/lib/x86_64-linux-gnu/libGLEW.so"

    # Initialize environment
    env = TimeLimit(CliffWalkerEnv(use_custom_step=True), max_episode_steps=5000)

    # Video writer settings
    video_path = "walker2d_video.mp4"
    frame_width = 640  # Adjust as needed
    frame_height = 480  # Adjust as needed
    fps = 30

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(video_path, fourcc, fps, (frame_width, frame_height))

    # Reset environment
    env.reset()

    try:
        for _ in range(1000):
            # Step environment
            action = env.action_space.sample()
            _, _, done, _ = env.step(action)

            # Render frame as RGB array
            frame = env.render(mode='rgb_array')

            # Resize frame to video dimensions
            frame = cv2.resize(frame, (frame_width, frame_height))

            # Convert RGB to BGR for OpenCV
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # Write frame to video
            video_writer.write(frame_bgr)

            if done:
                env.reset()
    except Exception as e:
        print(f"Error during rendering or video writing: {e}")
    finally:
        video_writer.release()
        env.close()

    print(f"Video saved to {video_path}")