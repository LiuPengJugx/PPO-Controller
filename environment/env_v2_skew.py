import gym
from gym.spaces import *
import numpy as np
from controller.db.conf import Conf
from controller.db.disk_io import DiskIo
from controller.db.workload import Workload
from controller.par_algorithm.scvp import Scvp


class EnvSkew2(gym.Env):
    file_path= '../data/data1/1500query-steam.csv'
    # hyper-parameters
    reward_threshold=0.1
    adjust_range_threshold=10
    judge_punishment=-0.05
    cost_weight=[1,1.5]
    def __init__(self):
        self.w = Workload(Conf.TABLE_ATTRIBUTE_NUM, self.file_path)
        self.model = Scvp

        self.last_time_point=0
        self.time_point=0
        self.done=False
        self.cur_par_schema=[[i for i in range(self.w.attr_num)]]
        self.io_cost=[]
        self.action_list = dict()
        self.last_affinity_sel_matrix=self.w.compute_affinity_matrix_consider_selectivity(0, self.last_time_point)
        self.temp_affinity_sel_matrix=np.copy(self.last_affinity_sel_matrix)
        self.temp_workload=[]
        self.current_affinity_matrix=self.w.compute_affinity_matrix(0,0)
        self.observation_space = Box(shape=(50,50), low=-3000, high=3000, dtype=np.int)
        self.action_space = Discrete(2)
        # how many steps this env has stepped
        self.steps = 0
        self.adjust_times=0
        self.ideal_val =0
        self.last_par_schema=[]
        self.state=self._get_state(self.last_affinity_sel_matrix,None)

    def step(self,action):
        assert self.action_space.contains(action)
        self.steps += 1
        self.time_point += 1
        print("Sample action:",action," Step:",self.steps)
        self.temp_affinity_sel_matrix = self.w.update_affinity_matrix_consider_selectivity(self.temp_affinity_sel_matrix,self.time_point)
        self.current_affinity_matrix=self.w.update_affinity_matrix(self.current_affinity_matrix, self.time_point)
        state=self._get_state(self.last_affinity_sel_matrix, self.temp_affinity_sel_matrix)
        time_diff = self.time_point - self.last_time_point
        self.temp_workload += self.w.load_sql_by_time_range(self.time_point, self.time_point + 1)
        if self.time_point>=self.w.sql_list[-1]['time']:
            io_cost = DiskIo.compute_cost(self.temp_workload, self.cur_par_schema, self.w.attrs_length)
            self.io_cost.append(io_cost)
            self.done=True
            return state, 0, self.done, {}
        if action==1:
            # if workload scale is zero, we could skip partitioning step.
            if len(self.temp_workload) == 0:
                return state, 0, self.done, {}
            # get partition schema and compute reward
            new_par_schema = self.model.partitioner(self.current_affinity_matrix, self.temp_workload, self.w.attrs_length)
            operation_cost = DiskIo.compute_repartitioning_cost(self.cur_par_schema, new_par_schema, self.w.attrs_length)
            reward, io_cost = self._get_reward(self.cur_par_schema, new_par_schema, self.temp_workload, operation_cost, time_diff)
            print(reward)
            if reward>=self.reward_threshold:
                self.io_cost.append(io_cost)
                self.last_par_schema=self.cur_par_schema.copy()
                self.cur_par_schema = new_par_schema
                self.last_time_point = self.time_point
                self.last_affinity_sel_matrix = np.copy(self.temp_affinity_sel_matrix)
                self.temp_affinity_sel_matrix = self.w.compute_affinity_matrix_consider_selectivity(0, 0)
                self.current_affinity_matrix = self.w.compute_affinity_matrix(0, 0)
                self.temp_workload = []
                self.adjust_times = 0
                # print("~~~~~~ action:", action, " , step:", self.steps, " , state:",state, " , reward:", reward, " ,done:", self.done)
                print("~~~~~~ time stage:", str([self.last_time_point,self.time_point]), " , step:", self.steps, " , reward:", reward, " ,done:", self.done)
                # self.action_list.append([self.time_point,1])
                self.action_list[self.time_point]=new_par_schema
                return state, reward, self.done, {}
            else:
                # reward为正，但值过小，可以理解为：此时可以有略微提升，但控制器认为仍需要维持现状.但也有可能是，分区算法考虑的负载范围较大，不适应后续的负载。
                if reward>=-self.reward_threshold:
                    self.adjust_times+=1
                    if self.adjust_times>=self.adjust_range_threshold:
                        self.io_cost.append(io_cost)
                        self.last_time_point = self.time_point
                        self.last_affinity_sel_matrix = np.copy(self.temp_affinity_sel_matrix)
                        self.temp_affinity_sel_matrix = self.w.compute_affinity_matrix_consider_selectivity(0, 0)
                        self.current_affinity_matrix = self.w.compute_affinity_matrix(0, 0)
                        self.temp_workload = []
                        self.adjust_times = 0
                # 添加分区判断的成本
                self.io_cost.append(operation_cost/3)
                # self.action_list.append([self.time_point, 0])
                return state, self.judge_punishment, self.done, {}
        else:
            return state, 0, self.done, {}

    def _get_reward(self,old_par_schema, new_par_schema, temp_workload, operation_cost, time_diff):
        front_cost = DiskIo.compute_cost(temp_workload, old_par_schema, self.w.attrs_length)
        after_cost = DiskIo.compute_cost(temp_workload, new_par_schema, self.w.attrs_length)
        io_cost = front_cost + operation_cost
        # 由于对未来负载情况未知，增加理想成本和操作成本的权重系数w1、w2
        w1, w2 = self.cost_weight[0], self.cost_weight[1]
        reward = ((front_cost - after_cost) * w1 - operation_cost * w2) / (front_cost * time_diff)
        return reward, io_cost

    def _get_reward2(self,old_par_schema,new_par_schema,temp_workload,operation_cost,time_diff):
        front_cost = DiskIo.compute_cost(temp_workload, old_par_schema, self.w.attrs_length)
        after_cost = DiskIo.compute_cost(temp_workload, new_par_schema, self.w.attrs_length)
        # temp_query_num=sum([sql.frequency for sql in temp_workload])
        std_cost=DiskIo.compute_cost(temp_workload, [[i] for i in range(self.w.attr_num)], self.w.attrs_length)
        # 计算偏斜度
        if self.ideal_val != 0:
            last_front_cost=DiskIo.compute_cost(temp_workload,self.last_par_schema,self.w.attrs_length)
            real_val=(last_front_cost-front_cost)/ std_cost
            skew_degree=real_val/self.ideal_val
        else:
            skew_degree=1
        self.ideal_val=(front_cost-after_cost)/ std_cost
        io_cost=front_cost + operation_cost
        # 由于对未来负载情况未知，增加理想成本和操作成本的权重系数w1、w2
        w1,w2=self.cost_weight[0],self.cost_weight[1]
        reward=((front_cost-after_cost)*w1-operation_cost*w2)*skew_degree/(front_cost*time_diff)
        return reward,io_cost

    def _get_state(self,last_affinity_sel_matrix,current_affinity_sel_matrix):
        if current_affinity_sel_matrix is None:
            return last_affinity_sel_matrix
        else:
            return last_affinity_sel_matrix-current_affinity_sel_matrix

    def reset(self, state=None):
        self.last_time_point = 0
        self.time_point = 0
        self.done=False
        self.cur_par_schema = [[i for i in range(self.w.attr_num)]]
        self.io_cost = []
        self.action_list = dict()
        self.temp_workload = []
        self.last_affinity_sel_matrix = self.w.compute_affinity_matrix_consider_selectivity(0, self.last_time_point)
        self.temp_affinity_sel_matrix = np.copy(self.last_affinity_sel_matrix)
        self.current_affinity_matrix = self.w.compute_affinity_matrix(0, 0)
        self.steps = 0
        self.adjust_times = 0
        self.ideal_val = 0
        self.last_par_schema = []
        if state is None:
            self.last_affinity_sel_matrix = self.w.compute_affinity_matrix_consider_selectivity(0, self.last_time_point)
            state = self._get_state(self.last_affinity_sel_matrix,None)
        return state

    def seed(self, seed=0):
        self.rng = np.random.RandomState(seed)
        return [seed]


    