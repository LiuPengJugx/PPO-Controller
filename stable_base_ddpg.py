import os
import numpy as np
from controller.environment.multenv_v2 import Multenv2
from stable_baselines3 import DDPG
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise


class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a pretrained (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).
    :param check_freq: (int)
    :param log_dir: (str) Path to the folder where the pretrained will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: (int)
    """
    def __init__(self, check_freq: int, log_dir: str, verbose=1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model')
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
          # Retrieve training reward
          x, y = ts2xy(load_results(self.log_dir), 'timesteps')
          if len(x) > 0:
              # Mean training reward over the last 100 episodes
              mean_reward = np.mean(y[-100:])
              if self.verbose > 0:
                print(f"Num timesteps: {self.num_timesteps}")
                print(f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}")

              # New best pretrained, you could save the agent here
              if mean_reward > self.best_mean_reward:
                  self.best_mean_reward = mean_reward
                  # Example for saving best pretrained
                  if self.verbose > 0:
                    print(f"Saving new best pretrained to {self.save_path}.zip")
                  self.model.save(self.save_path)
        return True

num_cpu=5
log_dir = "stable_pretrained_model/ddpg3000_monitor/"
os.makedirs(log_dir, exist_ok=True)
env=Multenv2()
# env=SubprocVecEnv([make_env(env, i) for i in range(num_cpu)])
# vec_env=make_vec_env(Multenv, n_envs=num_cpu,monitor_dir=log_dir)
env=Monitor(env,log_dir)
callback = SaveOnBestTrainingRewardCallback(check_freq=100, log_dir=log_dir)
# Create action noise because TD3 and DDPG use a deterministic policy
n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.2 * np.ones(n_actions))
# pretrained = DDPG("MlpPolicy", env, verbose=0,action_noise=action_noise)
# pretrained.learn(total_timesteps=10000,callback=callback)
# # Save the agent
# pretrained.save("stable_pretrained_model/ddpg_controller")
# del pretrained  # delete trained pretrained to demonstrate loading
model = DDPG.load("stable_pretrained_model/ddpg3000_monitor/best_model.zip")
obs = env.reset()
epoch=0
total_reward=[]
for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    total_reward.append(reward)
    if done:
        print(f"The total reward : {sum(total_reward)}")
        print('io_cost : ', env.io_cost)
        print('action list : ', env.action_list)
        print(f"Average query cost: {sum(env.io_cost) / sum([sql['feature'].frequency for sql in env.w.sql_list])}")
        epoch+=1
        obs = env.reset()
        if  epoch==1:break
env.close()

# from stable_baselines3.common import results_plotter
# #Helper from the library
# results_plotter.plot_results([log_dir], 10000, results_plotter.X_TIMESTEPS, "A2C Controller")


# DDPG
# Steps:1e4
# reward: -3.2429015111010715
# io_cost :  [9054176.0, 68448.0, 12980.0]
# action list :  [[4, 15], [15, 16], [16, 17]]
# Average query cost: 271.0298751001276


