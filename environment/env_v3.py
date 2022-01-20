import gym
from gym.spaces import *
import numpy as np
from controller.db.conf import Conf
from controller.db.disk_io import DiskIo
from controller.db.workload import Workload
from controller.par_algorithm.scvp import Scvp


class Env3(gym.Env):
    file_path= '../data/data1/10000query-steam.csv'
    start_time=1500
    def __init__(self):
        self.w = Workload(Conf.TABLE_ATTRIBUTE_NUM, self.file_path)
        self.w.prune_sql_list(1500,2001)
        # self.w.prune_sql_list(0,1500)
        self.model = Scvp
        self.last_time_point=self.start_time
        self.time_point=self.start_time
        self.done=False
        self.cur_par_schema=[[i for i in range(self.w.attr_num)]]
        self.io_cost=[]
        self.action_list = dict()
        self.cost_list=dict()
        self.temp_workload=[]
        self.last_affinity_sel_matrix=self.w.compute_affinity_matrix_consider_selectivity(0, 0)
        self.temp_affinity_sel_matrix=np.copy(self.last_affinity_sel_matrix)
        self.current_affinity_matrix=self.w.compute_affinity_matrix(0,0)
        self.observation_space = Box(shape=(50,50), low=-3000, high=3000, dtype=np.int)
        self.action_space = Discrete(2)
        # how many steps this env has stepped
        self.steps = 0
        self.adjust_times = 0
        self.ideal_val = 0
        self.last_par_schema = []
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
        self.temp_workload += self.w.load_sql_by_time_range(self.time_point,self.time_point+1)
        if self.time_point>=self.w.sql_list[-1]['time']:
            io_cost = DiskIo.compute_cost(self.temp_workload, self.cur_par_schema, self.w.attrs_length)
            self.io_cost.append(io_cost)
            self.done=True
            print(self.done)
            return state, 0, self.done, {}
        if action==1:
            # get partition schema and compute reward
            next_workload = self.w.load_sql_by_time_range(self.time_point+1,self.time_point+1+self.time_point-self.last_time_point)
            # if workload scale is zero, we could skip partitioning step.
            if len(self.temp_workload)==0 or len(next_workload)== 0:
                return state, 0, self.done, {}
            new_par_schema = self.model.partitioner(self.current_affinity_matrix, self.temp_workload, self.w.attrs_length)
            # new_par_schema = self.pretrained.partitioner(self.current_affinity_matrix, next_workload, self.w.attrs_length)
            operation_cost = DiskIo.compute_repartitioning_cost(self.cur_par_schema, new_par_schema, self.w.attrs_length)
            reward, io_cost = self._get_reward(self.cur_par_schema, new_par_schema, self.temp_workload,next_workload, operation_cost, time_diff)
            print(reward)
            # reward=1
            # if reward>=0.1 or operation_cost==0:
            if reward>0 or operation_cost==0:
                self.io_cost.append(io_cost)
                self.cost_list[self.time_point]=io_cost
                self.last_par_schema = self.cur_par_schema.copy()
                self.cur_par_schema = new_par_schema
                self.last_time_point = self.time_point
                self.last_affinity_sel_matrix = np.copy(self.temp_affinity_sel_matrix)
                self.temp_affinity_sel_matrix = self.w.compute_affinity_matrix_consider_selectivity(0, 0)
                self.current_affinity_matrix = self.w.compute_affinity_matrix(0, 0)
                self.temp_workload = []
                self.adjust_times = 0
                # print("~~~~~~ action:", action, " , step:", self.steps, " , state:",state, " , reward:", reward, " ,done:", self.done)
                print("~~~~~~ time stage:", str([self.last_time_point,self.time_point]), " , step:", self.steps, " , reward:", reward, " ,done:", self.done)
                self.action_list[self.time_point-1]=new_par_schema
                return state, reward, self.done, {}
            else:
                # if reward>-0.1:
                #     self.adjust_times+=1
                #     if self.adjust_times>=10:
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
                # return state, -0.005, self.done, {}
                return state, reward, self.done, {}
        else:
            return state, 0, self.done, {}

    # 开天眼的奖励计算
    def _get_reward(self,old_par_schema,new_par_schema,temp_workload,next_workload,operation_cost,time_diff):
        original_cost = DiskIo.compute_cost(next_workload,old_par_schema,self.w.attrs_length)
        update_cost = DiskIo.compute_cost(next_workload,new_par_schema,self.w.attrs_length)
        io_cost=DiskIo.compute_cost(temp_workload,old_par_schema,self.w.attrs_length)+operation_cost
        # io_cost=DiskIo.compute_cost(temp_workload,old_par_schema,self.w.attrs_length)
        reward=(original_cost-update_cost-operation_cost)/(original_cost*time_diff)
        return reward,io_cost

    # 增加偏斜度的奖励
    def _get_reward2(self,old_par_schema,new_par_schema,temp_workload,next_workload,operation_cost,time_diff):
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
        w1,w2=1,2
        reward=((front_cost-after_cost)*w1-operation_cost*w2)*skew_degree/(front_cost*time_diff)
        return reward,io_cost

    # 正常奖励
    def _get_reward3(self,old_par_schema,new_par_schema,temp_workload,next_workload,operation_cost,time_diff):
        front_cost = DiskIo.compute_cost(temp_workload, old_par_schema, self.w.attrs_length)
        after_cost = DiskIo.compute_cost(temp_workload, new_par_schema, self.w.attrs_length)
        io_cost = front_cost + operation_cost
        # 由于对未来负载情况未知，增加理想成本和操作成本的权重系数w1、w2
        # w1, w2 = 1, 2
        w1, w2 = 1, 1.5
        reward = ((front_cost - after_cost) * w1 - operation_cost * w2) / (front_cost * time_diff)
        return reward, io_cost

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
        self.io_cost = []
        self.action_list=dict()
        self.cost_list = dict()
        self.temp_workload = []
        self.steps = 0
        self.adjust_times = 0
        self.ideal_val = 0
        self.last_par_schema = []
        self.last_affinity_sel_matrix = self.w.compute_affinity_matrix_consider_selectivity(0, 0)
        self.temp_affinity_sel_matrix = np.copy(self.last_affinity_sel_matrix)
        self.current_affinity_matrix = self.w.compute_affinity_matrix(0, 0)
        if state is None:
            state = self._get_state(self.last_affinity_sel_matrix,None)
        return state

    def seed(self, seed=0):
        self.rng = np.random.RandomState(seed)
        return [seed]


    