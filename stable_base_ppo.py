import time
import torch
from stable_baselines3 import PPO
import numpy as np
from controller.db.workload import Workload
from controller.environment.env_v2 import Env2
from controller.environment.env_v2_skew import EnvSkew2
from controller.environment.env_v2_2600 import Env2600
from controller.environment.env_v2_mask import EnvMask
from controller.environment.env_v2_mask_new import EnvMaskNew
from controller.environment.env_v2_par_mask import EnvParMask
from controller.environment.env_v2_2600_speedup import Env2600Speedup
from controller.environment.env_v2_state import EnvState
from controller.environment.env_v2_mask_curve import EnvMaskCurve
from controller.environment.env_v2_classifier import EnvWithClassifier
from controller.environment.env_v4 import Env4
from pretrained.workload_predictor import WorkloadClassifier
import multiprocessing as mul
from controller.par_algorithm.MyThread import MyThread
class PpoController:
    action_list=dict()

    @staticmethod
    def train_ppo_model(env,reward_threshold,saved_model_path,queue):
        env.reward_threshold = reward_threshold
        # model = PPO("MlpPolicy", env, verbose=1,device='cuda:4',tensorboard_log="./ppo_controller_tensorboard_env2/env4/")
        # # Run Command: tensorboard --logdir ./ppo_controller_tensorboard_env2/env4/  --bind
        # model.learn(total_timesteps=2e5)
        # model.save(saved_model_path)
        # del model
        results = dict()
        model = PPO.load(saved_model_path,device='cuda:4')
        obs = env.reset()
        epoch = 0
        total_reward = []
        # Record the proportion of awards that are positive
        total_actions = 0
        good_actions = 0
        reward_list = []
        for i in range(5000):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            reward_list.append(reward)
            if reward != 0: total_actions += 1
            if reward > 0: good_actions += 1
            total_reward.append(reward)
            if done:
                result = {
                          'reward': sum(total_reward),
                          'avg_cost': sum(env.io_cost) / sum([sql['feature'].frequency for sql in env.w.sql_list]),
                          'action_ratio': [len(env.action_list.keys()), round(good_actions / total_actions, 3)],
                          'rep_cost': sum(env.rep_cost) / sum([sql['feature'].frequency for sql in env.w.sql_list])}
                results[reward_threshold]=result
                print(f"The total reward : {result['reward']}")
                print(env.cost_time_map['cost'])
                print(env.cost_time_map['cdf'])
                print('action list : ', env.action_list.keys())
                print(f"Average query cost: {result['avg_cost']}")
                print(f"Average repartition cost: {result['rep_cost']}")
                print(reward_list)
                epoch += 1
                obs = env.reset()
                if epoch == 1: break
        print(results)
        env.close()
        queue.put(results)

    @staticmethod
    def train_model_diff_reward_threshold(wLoad):
        reward_threshold_list=np.linspace(0.05,0.2,7)
        queue = mul.Queue()
        jobs = []
        chunk_result=list()
        for reward_threshold in reward_threshold_list:
            reward_threshold=round(reward_threshold,3)
            env = Env4(wLoad)
            process = mul.Process(target=PpoController.train_ppo_model, args=(env,reward_threshold,"stable_pretrained_model/ppo_controller_v4_2800_"+str(reward_threshold),queue))
            process.start()
            jobs.append(process)
        for idx,_ in enumerate(jobs):
            chunk_result.append(queue.get())
        # for process in jobs: process.join()
        print(chunk_result)

    def repartition2(self,initial_par, wLoad, **kwargs):
        env = Env2()
        env.reward_threshold = 0.1
        # pretrained = PPO("MlpPolicy", env, verbose=1)
        # # pretrained = PPO("MlpPolicy", env, verbose=1,tensorboard_log="./ppo_controller_tensorboard_env2/")
        # # Run Command: tensorboard --logdir ./ppo_controller_tensorboard_env2/
        # pretrained.learn(total_timesteps=2e4)
        # pretrained.save("stable_pretrained_model/ppo_controller_v2_1500")
        # del pretrained
        model = PPO.load("stable_pretrained_model/ppo_controller_v2_1500")
        # pretrained = PPO.load("stable_pretrained_model/ppo_controller_v2_skew")
        obs = env.reset()
        epoch = 0
        total_reward = []
        for i in range(1000):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            total_reward.append(reward)
            if done:
                result = {'reward': sum(total_reward),
                          'avg_cost': sum(env.io_cost) / sum([sql['feature'].frequency for sql in env.w.sql_list])}
                print(f"The total reward : {result['reward']}")
                print('io_cost : ', env.io_cost)
                print('action list : ', env.action_list)
                self.action_list=env.action_list
                print(f"Average query cost: {result['avg_cost']}")
                epoch += 1
                obs = env.reset()
                if epoch == 1: break
        env.close()
    def repartition3(self,initial_par, wLoad, **kwargs):
        env = EnvSkew2()
        env.reward_threshold = 0.1
        # pretrained = PPO("MlpPolicy", env, verbose=1)
        # # pretrained = PPO("MlpPolicy", env, verbose=1,tensorboard_log="./ppo_controller_tensorboard_env2/")
        # # Run Command: tensorboard --logdir ./ppo_controller_tensorboard_env2/ --bind_all
        # pretrained.learn(total_timesteps=2e4)
        # pretrained.save("stable_pretrained_model/ppo_controller_v2_1500")
        # del pretrained
        # pretrained = PPO.load("stable_pretrained_model/ppo_controller_v2_1500")
        model = PPO.load("stable_pretrained_model/ppo_controller_v2_skew")
        obs = env.reset()
        epoch = 0
        total_reward = []
        for i in range(1000):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            total_reward.append(reward)
            if done:
                result = {'reward': sum(total_reward),
                          'avg_cost': sum(env.io_cost) / sum([sql['feature'].frequency for sql in env.w.sql_list])}
                print(f"The total reward : {result['reward']}")
                print('io_cost : ', env.io_cost)
                print('action list : ', env.action_list)
                self.action_list=env.action_list
                print(f"Average query cost: {result['avg_cost']}")
                epoch += 1
                obs = env.reset()
                if epoch == 1: break
        env.close()

    def repartition1(self, initial_par, wLoad, **kwargs):
        for i in range(10000):
            # time.sleep(2)
            print(i)
        return 'over'

    def repartition(self,initial_par, wLoad, **kwargs):
        # env = EnvSkew4000(wLoad)
        # env = EnvWithClassifier(wLoad)
        # env.reward_threshold = 0.1
        # pretrained = PPO("MlpPolicy", env, verbose=1,tensorboard_log="./ppo_controller_tensorboard_env2/")
        # pretrained.learn(total_timesteps=5e4)
        # pretrained.save("stable_pretrained_model/ppo_controller_v2_4000_copy")
        # Run Command: tensorboard --logdir ./ppo_controller_tensorboard_env2/ --bind_all
        # del pretrained
        env = Env4(wLoad)
        env.reward_threshold = 0.1
        model = PPO.load("stable_pretrained_model/ppo_controller_v4_2800_best",device='cuda:4')
        obs = env.reset()
        epoch = 0
        total_reward = []
        # Record the proportion of awards that are positive
        total_actions=0
        good_actions=0
        reward_list=[]
        result=dict()
        for i in range(5000):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            reward_list.append(reward)
            if reward!=0:total_actions+=1
            if reward>0:good_actions+=1
            total_reward.append(reward)
            if done:
                result = {'reward': sum(total_reward),
                          'avg_cost': sum(env.io_cost) / sum([sql['feature'].frequency for sql in env.w.sql_list]),
                          'action_ratio':[len(env.action_list.keys()),round(good_actions/total_actions,3),total_actions],
                          'rep_cost':sum(env.rep_cost)/sum([sql['feature'].frequency for sql in env.w.sql_list])}
                          # 'optimize_time':env.optimize_time}
                print(f"The total reward : {result['reward']}")
                print(env.cost_time_map['cost'])
                print(env.cost_time_map['cdf'])
                print('action list : ', env.action_list.keys())
                self.action_list = env.action_list
                print(f"Average query cost: {result['avg_cost']}")
                print(f"Average repartition cost: {result['rep_cost']}")
                # print(f"Optimize Time : {env.optimize_time}")
                print(reward_list)
                epoch += 1
                if epoch == 1: break
        env.close()
        return result


if __name__=="__main__":
    torch.multiprocessing.set_start_method('spawn')
    ppo=PpoController()
    test_result=dict()
    task_list=[]
    workload_dict = {
        'data1': [1500, 3000, 1300, 4000],
        'data2': [1200, 1350],
        'data3': [1600, 2600]
    }
    for w_path in workload_dict.keys():
        for query_num in workload_dict[w_path]:
            wLoad = Workload(50, f'data/{w_path}/{query_num}query-steam.csv')
            # ppo.train_model_diff_reward_threshold(wLoad)
            task=MyThread(ppo.repartition,(None, wLoad))
            task.start()
            task_list.append(task)
    for i,task in enumerate(task_list):
        test_result[i]=task.get_result()
    print(test_result)