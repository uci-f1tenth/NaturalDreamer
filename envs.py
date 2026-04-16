import gymnasium as gym
import numpy as np

def getEnvProperties(env):
    assert isinstance(env.action_space, gym.spaces.Box), "Sorry, supporting only continuous action space for now"
    observationShape = env.observation_space.shape
    actionSize = env.action_space.shape[0]
    actionLow = env.action_space.low.tolist()
    actionHigh = env.action_space.high.tolist()
    return observationShape, actionSize, actionLow, actionHigh

class GymPixelsProcessingWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        observationSpace = self.observation_space
        newObsShape = observationSpace.shape[-1:] + observationSpace.shape[:2]
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=newObsShape, dtype=np.float32)

    def observation(self, observation):
        observation = np.transpose(observation, (2, 0, 1))/255.0
        return observation
    
class CleanGymWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        return obs, reward, done

    def reset(self, seed=None):
        obs, info = self.env.reset(seed=seed)
        return obs


class RustoracerWrapper:
    """Adapter that exposes Rustoracer's vectorized gym environment through the
    simple single-env interface that NaturalDreamer expects:
      reset() -> obs
      step(action) -> (obs, reward, done)
      render() -> rgb_array

    Also normalizes observations with running mean/std to stabilize training,
    because raw sensor values live on very different scales."""

    def __init__(self, yaml_path, max_steps=10000):
        from rustoracerpy import RustoracerEnv

        self._env = RustoracerEnv(
            yaml=yaml_path,
            num_envs=1,
            max_steps=max_steps,
            render_mode="rgb_array",
        )

        obs_dim = self._env.single_observation_space.shape[0]
        self.observation_space = gym.spaces.Box(
            low=-10.0, high=10.0, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = self._env.single_action_space

        # Running statistics for online observation normalization (Welford).
        self._obs_mean = np.zeros(obs_dim, dtype=np.float64)
        self._obs_var = np.ones(obs_dim, dtype=np.float64)
        self._obs_count = 1e-4

    def _update_stats(self, obs):
        """Welford online update for running mean and variance."""
        self._obs_count += 1
        delta = obs - self._obs_mean
        self._obs_mean += delta / self._obs_count
        self._obs_var += (
            delta * (obs - self._obs_mean) - self._obs_var
        ) / self._obs_count

    def _normalize(self, obs):
        """Standardize to roughly zero mean / unit variance and clip outliers."""
        obs = np.nan_to_num(obs, nan=0.0, posinf=10.0, neginf=-10.0)
        normalized = (obs - self._obs_mean) / np.sqrt(self._obs_var + 1e-8)
        return np.clip(normalized, -10.0, 10.0).astype(np.float32)

    def reset(self, seed=None):
        obs_batch, _info = self._env.reset(seed=seed)
        obs = obs_batch[0]
        self._update_stats(obs)
        return self._normalize(obs)

    def step(self, action):
        action_batch = np.array([action], dtype=np.float64)
        obs_batch, rewards, terminated, truncated, _info = self._env.step(
            action_batch
        )
        obs = obs_batch[0]
        # Clip reward to prevent -100 crash penalty from dominating the buffer
        # and destabilizing the reward/critic models. Preserves sign/direction.
        reward = float(np.clip(rewards[0], -5.0, 5.0))
        done = bool(terminated[0]) or bool(truncated[0])
        self._update_stats(obs)
        return self._normalize(obs), reward, done

    def render(self):
        """Return an RGB array for NaturalDreamer's video-saving code path."""
        return self._env.render()