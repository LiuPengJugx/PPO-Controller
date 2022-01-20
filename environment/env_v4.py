import time
import gym
from gym.spaces import *
import numpy as np
from controller.db.conf import Conf
from controller.db.disk_io import DiskIo
from controller.db.workload import Workload
from controller.par_algorithm.scvp import Scvp
from controller.util import Util
from controller.visualization.workload_data_collector import partition_by_wide_workload
import torch
class Env4(gym.Env):
    # hyper-parameters
    reward_threshold=0.1
    cost_weight=[1,1]
    def __init__(self,wLoad):
        self.w = wLoad
        self.model = Scvp
        self.start_time = self.w.sql_list[0]['time'] - 1
        self.last_time_point=self.start_time
        self.time_point=self.start_time
        self.done=False
        self.cur_par_schema=[[i for i in range(self.w.attr_num)]]
        self.io_cost=[]
        self.cost_time_map={"cost":list(),"cdf":list()}
        self.rep_cost=[]
        self.action_list = dict()
        self.temp_mask_affinity_matrix =np.array(([[0] * self.w.attr_num]) * self.w.attr_num)
        self.workload=dict()
        self.observation_space = Box(shape=(50,50), low=0, high=4000, dtype=np.int)
        self.action_space = Discrete(2)
        # how many steps this env has stepped
        self.steps = 0
        self.adjust_times=0
        # self.optimize_time=0
        self.state=self._get_state(self.time_point+1)
    def step(self,action):
        assert self.action_space.contains(action)
        self.steps += 1
        self.time_point += 1
        print("Sample action:",action," Step:",self.steps)
        state=self._get_state(self.time_point+1)
        self.workload[self.time_point] = self.w.load_sql_by_time_range(self.time_point, self.time_point + 1)
        if self.time_point>self.w.sql_list[-1]['time']:
            time_diff_workload=[]
            for key in sorted(self.workload.keys(), reverse=True):
                if key >= self.last_time_point + 1:
                    time_diff_workload += self.workload[key]
                else:
                    break
            io_cost = DiskIo.compute_cost(time_diff_workload, self.cur_par_schema, self.w.attrs_length)
            self.io_cost.append(io_cost)
            print("average_cost:",sum(self.io_cost) / sum([sql['feature'].frequency for sql in self.w.sql_list]))
            self.done=True
            return state, 0, self.done, {}
        if action==1 or self.time_point==self.w.sql_list[0]['time']:
            # if workload scale is zero, we could skip partitioning step.
            if len(self.workload[self.time_point]) == 0:
                self.temp_mask_affinity_matrix = state
                return state, 0, self.done, {}
            # get partition schema and compute reward
            new_par_schema,cluster_start_time = partition_by_wide_workload(self.workload,self.w)
            temp_cur_par_schema=self.cur_par_schema.copy()
            if len(temp_cur_par_schema) == 1 and len(temp_cur_par_schema[0]) == 50:
                temp_cur_par_schema=[[num] for num in range(50)]
            for par in temp_cur_par_schema:
                temp_par=par.copy()
                for par_new in new_par_schema:
                    if Util.list_solved_list(par_new,temp_par):
                        [temp_par.remove(attr) for attr in par_new if attr in temp_par]
                if len(temp_par)==len(par) or len(temp_par)==1:new_par_schema.append(temp_par)
                else:
                    [new_par_schema.append([attr]) for attr in temp_par]

            operation_cost = DiskIo.compute_repartitioning_cost(self.cur_par_schema, new_par_schema, self.w.attrs_length)
            reward, io_cost = self._get_reward(self.cur_par_schema, new_par_schema, operation_cost,cluster_start_time)
            # self.optimize_time +=time.time()-time0
            print(reward)
            if reward>=self.reward_threshold  or self.time_point==self.w.sql_list[0]['time']:
                # if classifier_flag: self.classifier_true_action[self.time_point] = reward
                # self.io_cost.append(io_cost+operation_cost)
                self.io_cost.append(io_cost)
                self.rep_cost.append(operation_cost)
                self.cur_par_schema = new_par_schema
                self.cost_time_map['cost'].append(f"({self.last_time_point+1} {self.time_point})->{io_cost}")
                self.cost_time_map['cdf'].append(f"{self.time_point}->{sum(self.io_cost)}")
                self.last_time_point = self.time_point
                self.temp_mask_affinity_matrix = np.array(([[0] * self.w.attr_num]) * self.w.attr_num)
                state = self._get_state(self.time_point + 1)
                self.adjust_times = 0
                print("~~~~~~ time stage:", str([self.last_time_point,self.time_point]), " , step:", self.steps, " , reward:", reward, " ,done:", self.done,' ,cost_blocks:',io_cost,' ,current par:',new_par_schema)
                self.action_list[self.time_point]=new_par_schema
            elif reward!=0:
                reward=(reward-self.reward_threshold*3/5)

            self.temp_mask_affinity_matrix=state
            # print("State:", np.sum(state, axis=0))
            return state, reward, self.done, {}
        else:
            self.temp_mask_affinity_matrix = state
            # print("State:", np.sum(state, axis=0))
            return state, 0, self.done, {}
    def _get_reward(self,old_par_schema, new_par_schema, operation_cost,cluster_start_time):
        time_diff_workload=[]
        temp_workload=[]
        stop_time=max(cluster_start_time,self.last_time_point+1)
        for key in sorted(self.workload.keys(), reverse=True):
            if key >=stop_time:
                temp_workload+=self.workload[key]
            if key >= self.last_time_point+1:
                time_diff_workload+=self.workload[key]
            if key <self.last_time_point+1 and key<stop_time:
                break
        front_cost = DiskIo.compute_cost(temp_workload, old_par_schema, self.w.attrs_length)
        after_cost = DiskIo.compute_cost(temp_workload, new_par_schema, self.w.attrs_length)
        io_cost = DiskIo.compute_cost(time_diff_workload, old_par_schema, self.w.attrs_length)
        w1, w2 = self.cost_weight[0], self.cost_weight[1]
        reward = ((front_cost - after_cost) * w1 - operation_cost * w2) / front_cost
        return reward, io_cost


    def _get_state(self,time):
        return self.w.mask_affinity_matrix_improvement(self.temp_mask_affinity_matrix,time,self.cur_par_schema)



    def reset(self, state=None):
        self.last_time_point = self.start_time
        self.time_point = self.start_time
        self.done=False
        self.cur_par_schema = [[i for i in range(self.w.attr_num)]]
        self.io_cost.clear()
        self.cost_time_map={"cost":list(),"cdf":list()}
        self.rep_cost.clear()
        self.action_list.clear()
        self.workload.clear()
        self.temp_mask_affinity_matrix = np.array(([[0] * self.w.attr_num]) * self.w.attr_num)
        self.steps = 0
        self.adjust_times = 0
        # self.optimize_time = 0
        if state is None:
            state = self._get_state(self.time_point+1)
        return state

    def seed(self, seed=0):
        return [seed]


    