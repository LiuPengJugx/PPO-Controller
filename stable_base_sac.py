from stable_baselines3 import SAC
from controller.environment.multenv_v2 import Multenv2
env = Multenv2()
# pretrained = SAC("MlpPolicy", env, verbose=1,learning_rate=1e-3)
# log_dir = "stable_pretrained_model/sac_3000_monitor/"
# pretrained.learn(total_timesteps=1e4,eval_freq=1000, n_eval_episodes=5, eval_log_path=log_dir)
# # Save the agent
# pretrained.save("stable_pretrained_model/sac_controller_v2")
# del pretrained  # delete trained pretrained to demonstrate loading

model = SAC.load("stable_pretrained_model/sac_controller_v2")
obs = env.reset()
epoch=0
total_reward=[]
for i in range(100):
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

# 3000query-steam.csv


# Steps:10000
# io_cost :
# action list :
# Average query cost:


# steps:10000
# The total reward : -0.29868847807270976
# io_cost :  [111680.0, 21860.0, 414514.0, 7463908.0]
# action list :  [[2, 4], [4, 6], [5, 10]]  or  [[-46, 4], [4, 6], [5, 10]]
# Average query cost: 237.69430682054173

