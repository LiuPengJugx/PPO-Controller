import os
import numpy as np
from controller.environment.env_v2 import Env2
from stable_baselines3 import A2C
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.env_util import make_vec_env


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
num_cpu=3
log_dir = "stable_pretrained_model/a2c3000_monitor/"
os.makedirs(log_dir, exist_ok=True)
# env=SubprocVecEnv([make_env(env, i) for i in range(num_cpu)])
vec_env=make_vec_env(Env2, n_envs=num_cpu,monitor_dir=log_dir)
# env=Monitor(env,log_dir)
callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=log_dir)
# pretrained = A2C("MlpPolicy", vec_env, verbose=0)
# pretrained.learn(total_timesteps=1e5,callback=callback)
# # Save the agent
# pretrained.save("stable_pretrained_model/a2c_controller")
# del pretrained  # delete trained pretrained to demonstrate loading
model = A2C.load("stable_pretrained_model/a2c3000_monitor/best_model.zip")
env=Env2()
obs = env.reset()
epoch=0
total_reward=[]
for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    total_reward.append(reward)
    if done:
        print(f"The total reward : {sum(total_reward)}")
        print('Length:',len(env.io_cost),' ; io_cost : ', env.io_cost)
        print('Length:',len(env.action_list),' ; action list : ', env.action_list)
        print(f"Average query cost: {sum(env.io_cost) / sum([sql['feature'].frequency for sql in env.w.sql_list])}")
        epoch+=1
        obs = env.reset()
        if  epoch==1:break
env.close()

# from stable_baselines3.common import results_plotter
# #Helper from the library
# results_plotter.plot_results([log_dir], 10000, results_plotter.X_TIMESTEPS, "A2C Controller")

# A2C
# steps:10000
# io_cost :  [6168910.0, 375495.0, 255790.0, 113357.0, 356065.0, 29835.0, 338790.0, 86699.0, 293214.0, 176250.0, 203633.0, 262168.0, 79184.0, 279308.0, 38690.0, 341084.0, 49978.0, 305357.0, 125280.0, 239733.0, 201434.0, 117671.0, 277141.0, 80120.0, 393878.0, 55582.0, 313373.0, 49473.0, 177082.0, 103254.0, 115159.0, 131613.0, 64100.0, 205203.0, 18963.0, 196397.0, 49703.0, 192121.0, 66613.0, 149780.0, 93365.0, 103837.0, 135512.0, 53922.0, 195865.0, 9940.0]
# action list :  [11, 22, 33, 44, 55, 66, 77, 88, 99, 110, 121, 132, 143, 154, 165, 176, 187, 198, 209, 220, 231, 242, 253, 264, 275, 286, 297, 308, 319, 330, 341, 352, 363, 374, 385, 396, 407, 418, 429, 440, 451, 462, 473, 484, 495, 506]
# Average query cost: 405.57546358107106


# A2C
# steps:50000
# action list : [2 4 6 8 .. 200]
# Average query cost:?????(未达到Done=True)

# Environment Version 2:
# A2C
# steps:5e4
# action list :  [3, 4, 5, 6, 7, 8 ... ]
# Average query cost: 256.9003708648568

# steps:5e5
# 权重为 1和1.5
# action list :  [3, 4, 5, 6, 7, 8 ... ]
# Average query cost: 253.93478712357216
# 权重为 1和2
# Length: 221  ; io_cost
# Length: 220 action list :  [3, 4, 5, 6, 7, 8 ... ]
# Average query cost: 252.57228897789645



# steps:1e5
# 权重为 1和1.5
# Length: 2  ; io_cost :  [33360, 8212534.0]
# Length: 2  ; action list :  [[3, 1], [4, 0]]
# Average query cost: 244.64898383029225


