import gym
from gym.spaces import *
import numpy as np
from controller.db.conf import Conf
from controller.db.disk_io import DiskIo
from controller.db.workload import Workload
from controller.par_algorithm.scvp import Scvp
from controller.util import Util
import math
class Multenv2(gym.Env):
    file_path='/home/liupengju/pycharmProjects/hbaselines/hbaselines/envs/controller/data/data1/3000query-steam.csv'
    def __init__(self):
        # 将时间分为间隔小的时间段
        self.w = Workload(Conf.TABLE_ATTRIBUTE_NUM, self.file_path)
        self.model = Scvp
        self.rng = np.random.RandomState(0)
        self.last_action=0
        self.time_point=self.last_action
        self.done=False
        affinity_matrix = self.w.compute_affinity_matrix(0, self.last_action)
        self.last_affinity_matrix=affinity_matrix
        self.last_schema=[[i for i in range(self.w.attr_num)]]
        self.cost_status=[0,0]
        self.io_cost=[]
        self.action_list = []
        self.observation_space = Box(shape=(101,50), low=0, high=1000, dtype=np.int)
        self.action_space = Box(shape=(2,),low=1,high=50,dtype=np.int)
        self.steps = 0
        self.state=self._get_state2(self.last_schema,affinity_matrix)

    def step(self,action_dict):
        print("Sample action: ",action_dict," ,Step:",self.steps+1)
        assert self.action_space.contains(action_dict)
        action=self.time_point+math.ceil(action_dict[0])
        range_start=action-math.ceil(action_dict[1])
        self.steps += 1
        if action>=self.w.sql_list[-1]['time']:
            temp_workload = self.w.load_sql_by_time_range(self.last_action, action)
            io_cost = DiskIo.compute_cost(temp_workload, self.last_schema, self.w.attrs_length)
            self.io_cost.append(io_cost)
            self.done=True
            return self._get_state2(self.last_schema, self.last_affinity_matrix), 0, self.done, {}
        # get partition schema and compute reward
        current_affinity_matrix = self.w.compute_affinity_matrix(range_start, action)
        refer_workload=self.w.load_sql_by_time_range(range_start, action)
        temp_workload = self.w.load_sql_by_time_range(self.last_action, action)
        self.time_point=action
        # if len(temp_workload)==0:
        #     self.last_action=action
        #     return self._get_state2(self.last_schema,current_affinity_matrix),0,self.done,{}
        # if len(refer_workload) == 0:
        #     return self._get_state2(self.last_schema, current_affinity_matrix), -0.01, self.done, {}
        if len(refer_workload)>0 and len(temp_workload)>0:
            new_par_schema = self.model.partitioner(current_affinity_matrix,refer_workload,self.w.attrs_length)
            operation_cost = DiskIo.compute_repartitioning_cost(self.last_schema,new_par_schema,self.w.attrs_length)
            time_diff=action-self.last_action
            reward,io_cost,front_frequency=self._get_reward(self.last_schema,new_par_schema,temp_workload,refer_workload,operation_cost,time_diff)
            print("Reward:",reward)
            if reward>=0.01:
                self.last_schema = new_par_schema
                last_action = self.last_action
                self.last_action = action
                self.last_affinity_matrix = current_affinity_matrix
                self.io_cost.append(io_cost)
                self.action_list.append([range_start, action])
                # # Update disk io cost status
                # self.cost_status[0] = io_cost
                # self.cost_status[1] = front_frequency
                state=self._get_state2(new_par_schema, current_affinity_matrix)
                print("time stage:", str([last_action,action]), " , step:", self.steps, " , state:",state, " , reward:", reward,
                      " ,done:", self.done)
                return state, reward, self.done, {}
            else:
                return self._get_state2(self.last_schema, current_affinity_matrix), -0.005, self.done, {}
        return self._get_state2(self.last_schema, self.last_affinity_matrix), 0, self.done, {}

    def _get_reward(self,old_par_schema,new_par_schema,temp_workload,refer_workload,operation_cost,time_diff):
        front_cost = DiskIo.compute_cost(temp_workload,old_par_schema,self.w.attrs_length)
        after_cost = DiskIo.compute_cost(temp_workload,new_par_schema,self.w.attrs_length)
        front_frequency = sum([item.frequency for item in temp_workload])
        io_cost=front_cost+operation_cost
        return (front_cost-after_cost-operation_cost) /(front_cost*time_diff),io_cost,front_frequency

    def _get_reward2(self,old_par_schema,new_par_schema,temp_workload,refer_workload,operation_cost,time_diff):
        front_cost = DiskIo.compute_cost(temp_workload,old_par_schema,self.w.attrs_length)
        front_frequency=sum([item.frequency for item in temp_workload])
        after_cost = DiskIo.compute_cost(refer_workload,new_par_schema,self.w.attrs_length)
        refer_frequency=sum([item.frequency for item in refer_workload])
        # avg_cost_per_query=self.cost_status[0]/self.cost_status[1] if self.cost_status[1]>0 else 0
        last_cost_per_query=front_cost/front_frequency
        forecast_cost_per_query=(after_cost+operation_cost)/refer_frequency
        # print(avg_cost_per_query,last_cost_per_query,forecast_cost_per_query)
        # 比例系数 proportionality coefficient
        # if avg_cost_per_query > 0:
        #     a,b=0.3,0.7
        # else: a,b=0,0.7
        # reward=a*((last_cost_per_query-avg_cost_per_query)/(time_diff))+b*(last_cost_per_query-forecast_cost_per_query)
        reward=last_cost_per_query-forecast_cost_per_query
        if self.done:
            io_cost=front_cost
        else: io_cost=front_cost+operation_cost
        return reward,io_cost,front_frequency

    def _get_state2(self,par_schema,affinity_matrix):
        # how to design correct state ?
        # Affinity matrix+partitions schema
        partition_vector=np.zeros([1,Conf.TABLE_ATTRIBUTE_NUM])
        tag=0
        standard_par=Util.partition_ordering(par_schema)
        for par in standard_par:
            for attr in par:partition_vector[0][attr]=tag
            tag+=1
        state=np.concatenate([affinity_matrix,partition_vector,self.last_affinity_matrix])
        return state

    def reset(self, state=None):
        self.last_action = 0
        self.done=False
        affinity_matrix = self.w.compute_affinity_matrix(0,self.last_action)
        initial_par = [[i for i in range(self.w.attr_num)]]
        self.last_schema=initial_par
        self.time_point = self.last_action
        self.cost_status=[0,0]
        self.io_cost=[]
        self.action_list = []
        self.steps = 0
        if state is None:
            state = self._get_state2(initial_par,affinity_matrix)
        return state

    def seed(self, seed=0):
        self.rng = np.random.RandomState(seed)
        return [seed]


    