import time

import gym
from gym.spaces import *
import numpy as np
from controller.db.conf import Conf
from controller.db.disk_io import DiskIo
from controller.db.workload import Workload
from controller.par_algorithm.scvp import Scvp


class EnvOpt(gym.Env):
    def __init__(self,wLoad):
        self.w = wLoad
        self.start_time=self.w.sql_list[0]['time']-1
        self.model = Scvp
        self.last_time_point=self.start_time
        self.time_point=self.start_time
        self.done=False
        self.cur_par_schema=[[i for i in range(self.w.attr_num)]]
        self.io_cost=[]
        self.rep_cost=[]
        self.action_list = dict()
        self.temp_workload=[]
        self.last_affinity_sel_matrix=self.w.compute_affinity_matrix_consider_selectivity(0, 0)
        self.temp_affinity_sel_matrix=np.copy(self.last_affinity_sel_matrix)
        self.observation_space = Box(shape=(50,50), low=-3000, high=3000, dtype=np.int)
        self.action_space = Discrete(2)
        # how many steps this env has stepped
        self.steps = 0
        self.ideal_val = 0
        self.last_par_schema = []
        self.optimize_time=0
        self.state=self._get_state(self.last_affinity_sel_matrix,None)

    def step(self,action):
        assert self.action_space.contains(action)
        self.steps += 1
        self.time_point += 1
        repar_time = self.time_point - 1
        print("Time point:",self.time_point," Sample action:",action," Step:",self.steps)
        time_diff = repar_time - self.last_time_point
        reward=0
        if self.time_point>=self.w.sql_list[-1]['time']:
            self.temp_workload += self.w.load_sql_by_time_range(self.time_point, self.time_point + 1)
            io_cost = DiskIo.compute_cost(self.temp_workload, self.cur_par_schema, self.w.attrs_length)
            self.io_cost.append(io_cost)
            self.done=True
            print(self.done)
            # return state, 0, self.done, {}
        else:
            if action==1:
                # get partition schema and compute reward
                next_workload = self.w.load_sql_by_time_range(repar_time+1,repar_time+repar_time-self.last_time_point+1)
                # if workload scale is zero, we could skip partitioning step.
                if len(self.temp_workload)>0 and len(next_workload)> 0:
                    self.current_affinity_matrix = self.w.compute_affinity_matrix(repar_time+1,repar_time+repar_time-self.last_time_point+1)
                    time0=time.time()
                    new_par_schema = self.model.partitioner(self.current_affinity_matrix, next_workload, self.w.attrs_length)
                    operation_cost = DiskIo.compute_repartitioning_cost(self.cur_par_schema, new_par_schema, self.w.attrs_length)
                    reward, io_cost = self._get_reward(self.cur_par_schema, new_par_schema, self.temp_workload, next_workload, operation_cost, time_diff)
                    self.optimize_time +=time.time()-time0
                    print(reward)
                    self.last_time_point = repar_time
                    self.temp_workload = []
                    self.last_affinity_sel_matrix = np.copy(self.temp_affinity_sel_matrix)
                    self.temp_affinity_sel_matrix = self.w.compute_affinity_matrix_consider_selectivity(0, 0)
                    if reward>=0.1 or operation_cost==0:
                        self.io_cost.append(io_cost)
                        self.rep_cost.append(operation_cost)
                        self.last_par_schema = self.cur_par_schema.copy()
                        self.cur_par_schema = new_par_schema
                        # print("~~~~~~ time stage:", str([self.last_time_point+1,self.time_point]), " , step:", self.steps, " , reward:", reward, " ,done:", self.done)
                        self.action_list[repar_time]=new_par_schema
                    else:
                        self.io_cost.append(io_cost)
                        # self.action_list[repar_time] = []
                elif len(self.temp_workload)==0:
                    #明明间隔内负载数量为0，说明state=0，如果调用action=1，给予一定的负值奖励
                    reward=-0.0005
        # 更新当前时刻亲和度矩阵和负载
        self.temp_affinity_sel_matrix = self.w.update_affinity_matrix_consider_selectivity(self.temp_affinity_sel_matrix, self.time_point)
        self.temp_workload += self.w.load_sql_by_time_range(self.time_point, self.time_point + 1)
        state = self._get_state(self.last_affinity_sel_matrix, self.temp_affinity_sel_matrix)
        return state, reward, self.done, {}

    # 开天眼的奖励计算
    def _get_reward(self,old_par_schema,new_par_schema,temp_workload,next_workload,operation_cost,time_diff):
        original_cost = DiskIo.compute_cost(next_workload,old_par_schema,self.w.attrs_length)
        update_cost = DiskIo.compute_cost(next_workload,new_par_schema,self.w.attrs_length)
        io_cost=DiskIo.compute_cost(temp_workload,old_par_schema,self.w.attrs_length)
        # io_cost=DiskIo.compute_cost(temp_workload,old_par_schema,self.w.attrs_length)
        reward=(original_cost-update_cost-operation_cost)/(original_cost*time_diff)
        return reward,io_cost
    def _get_state(self,last_affinity_sel_matrix,current_affinity_sel_matrix):
        if current_affinity_sel_matrix is None:
            return last_affinity_sel_matrix
        else:
            return last_affinity_sel_matrix-current_affinity_sel_matrix

    def reset(self, state=None):
        self.last_time_point = self.start_time
        self.time_point = self.start_time
        self.done=False
        self.cur_par_schema = [[i for i in range(self.w.attr_num)]]
        self.io_cost.clear()
        self.rep_cost.clear()
        self.action_list=dict()
        self.temp_workload.clear()
        self.steps = 0
        self.ideal_val = 0
        self.optimize_time = 0
        self.last_par_schema.clear()
        self.last_affinity_sel_matrix = self.w.compute_affinity_matrix_consider_selectivity(0, 0)
        self.temp_affinity_sel_matrix = np.copy(self.last_affinity_sel_matrix)
        if state is None:
            state = self._get_state(self.last_affinity_sel_matrix,None)
        return state

    def seed(self, seed=0):
        self.rng = np.random.RandomState(seed)
        return [seed]


    