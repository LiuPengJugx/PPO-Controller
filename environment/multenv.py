import gym
from gym.spaces import *
import numpy as np
from controller.db.conf import Conf
from controller.db.disk_io import DiskIo
from controller.db.workload import Workload
from controller.par_algorithm.scvp import Scvp
from controller.util import Util
class Multenv(gym.Env):
    file_path= '../data/data1/3000query-steam.csv'
    def __init__(self):
        # 将时间分为间隔小的时间段 
        # self.observation_space=Dict({'workload_id':Discrete(100),'cur_par_id':Discrete(100)})
        # x=current workload y=partition schema
        self.w = Workload(Conf.TABLE_ATTRIBUTE_NUM, self.file_path)
        self.model = Scvp
        self.last_action=0
        self.done=False
        affinity_matrix = self.w.compute_affinity_matrix(0, self.last_action)
        initial_par = [[i for i in range(self.w.attr_num)]]
        self.last_schema=initial_par
        self.cost_status=[0,0]
        self.io_cost=[]
        self.action_list = []
        # self.observation_space = Box(shape=(Conf.TABLE_ATTRIBUTE_NUM+2,Conf.TABLE_ATTRIBUTE_NUM), low=0, high=1000, dtype=np.int)
        self.observation_space = Box(shape=(2,), low=0, high=100, dtype=np.int)
        self.action_space = Box(shape=(2,),low=1,high=50,dtype=np.int)
        # self.action_space = Box(shape=(2,self.w.sql_list[-1]['time']+1),low=0,high=1,dtype=np.int)
        # self.action_space = MultiDiscrete([self.w.sql_list[-1]['time']+1,self.w.sql_list[-1]['time']+1])
        # how many steps this env has stepped
        self.steps = 0
        self.rng = np.random.RandomState(0)
        # self.state = self._get_state(self.last_action)
        self.state=self._get_state2(initial_par,affinity_matrix)

    def step(self,action_dict):
        print("Sample action: ",action_dict," ,Step:",self.steps+1)
        assert self.action_space.contains(np.round(action_dict))
        action=self.last_action+round(action_dict[0])
        range_start=round(action_dict[1])
        if range_start>action: range_start=action
        self.action_list.append([action,range_start])
        self.steps += 1
        if action>=self.w.sql_list[-1]['time']:
            self.done=True
        # get partition schema and compute reward
        affinity_matrix = self.w.compute_affinity_matrix(range_start, action)
        refer_workload=self.w.load_sql_by_time_range(range_start, action)
        temp_workload = self.w.load_sql_by_time_range(self.last_action, action)
        old_par_schema=self.last_schema
        if len(temp_workload)==0 or len(refer_workload) == 0:
            self.last_action=action
            return self._get_state2(old_par_schema,affinity_matrix),-0.01,self.done,{}
            # return self._get_state(action),0,self.done,{}
        new_par_schema = self.model.partitioner(affinity_matrix,refer_workload,self.w.attrs_length)
        operation_cost = DiskIo.compute_repartitioning_cost(old_par_schema,new_par_schema,self.w.attrs_length)
        time_diff=action-self.last_action
        reward,io_cost,front_frequency=self._get_reward(old_par_schema,new_par_schema,temp_workload,operation_cost,time_diff)
        # print(self.last_action, '  ', action,' ',reward,' ',operation_cost)
        # if reward>0:
        self.last_schema=new_par_schema
        self.last_action = action
        self.io_cost.append(io_cost)
        # Update disk io cost status
        self.cost_status[0] = io_cost
        self.cost_status[1] = front_frequency
        print("action:", action, " , step:", self.steps, " , state:", self._get_state2(new_par_schema,affinity_matrix), " , reward:", reward,
              " ,done:", self.done)
        # else:
        #     if self.done: self.io_cost.append(io_cost)
        #     return self._get_state2(old_par_schema, affinity_matrix),-0.01,self.done,{}
        return self._get_state2(new_par_schema,affinity_matrix), reward, self.done, {}


    def _get_reward(self,old_par_schema,new_par_schema,temp_workload,operation_cost,time_diff):
        front_cost = DiskIo.compute_cost(temp_workload,old_par_schema,self.w.attrs_length)
        after_cost = DiskIo.compute_cost(temp_workload,new_par_schema,self.w.attrs_length)
        front_frequency = sum([item.frequency for item in temp_workload])
        if self.done:
            io_cost=front_cost
        else: io_cost=front_cost+operation_cost
        # sum([item.frequency for item in temp_workload])
        return (front_cost-after_cost-operation_cost) /(front_cost*time_diff),io_cost,front_frequency

    def _get_reward2(self,old_par_schema,new_par_schema,temp_workload,refer_workload,operation_cost):
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
        return [self.steps, self.last_action]

    def _get_state(self,par_schema,affinity_matrix):
        # how to design correct state ?
        # Affinity matrix+partitions schema
        onehot_vector=np.zeros([1,Conf.TABLE_ATTRIBUTE_NUM])
        step_vector=np.ones([1,Conf.TABLE_ATTRIBUTE_NUM])*self.steps
        tag=0
        standard_par=Util.partition_ordering(par_schema)
        for par in standard_par:
            for attr in par:onehot_vector[0][attr]=tag
            tag+=1
        state=np.concatenate([affinity_matrix,onehot_vector,step_vector])
        return state
    # def seed(self, seed=None):
    #     return [seed]

    def reset(self, state=None):
        # reset 'done' 'state'
        self.last_action = 0
        # print("开始时间：",self.last_action)
        self.done=False
        affinity_matrix = self.w.compute_affinity_matrix(0,self.last_action)
        initial_par = [[i for i in range(self.w.attr_num)]]
        self.last_schema=initial_par
        self.cost_status=[0,0]
        # self.io_cost = []
        # self.action_list = []
        self.steps = 0
        if state is None:
            # state = self._get_state(self.last_action)
            state = self._get_state2(initial_par,affinity_matrix)
        return state

    def seed(self, seed=0):
        self.rng = np.random.RandomState(seed)
        return [seed]


    