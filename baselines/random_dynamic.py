import time

import numpy as np
from controller.par_algorithm.scvp import Scvp
from controller.baselines.algo import Algorithm
from controller.db.disk_io import DiskIo
class RandomDynamic(Algorithm):
    def partition(self, windowsql, wLoad, cur_par_schema):
        windowsql = windowsql.tolist()
        min_schema=cur_par_schema
        schema=Scvp.partitioner(wLoad.compute_affinity_matrix_by_sqls(windowsql),windowsql,wLoad.attrs_length)
        init_cost = DiskIo.compute_cost(windowsql, cur_par_schema, wLoad.attrs_length)
        cost = DiskIo.compute_cost(windowsql, schema, wLoad.attrs_length)
        if cost<init_cost:
            min_schema=schema
        return min_schema,init_cost-cost

    def repartition(self, initial_par, wLoad):
        total_blocks = 0
        total_rep_blocks = 0
        collector=np.array([])
        self.action_list = dict()
        cur_par_schema = initial_par
        self.optimize_time=0
        total_actions=0
        true_actions=0
        isFirst=True
        begin_time=wLoad.sql_list[0]['time']


        action_map = [8,40,70,130,180,270]
        for idx,sql in enumerate(wLoad.sql_list):
            if collector.shape[0]>=2:
                if (sql['time'] in action_map) or (sql['time']>=begin_time+2 and isFirst):
                    if sql['time'] in action_map:action_map.remove(sql['time'])
                    isFirst=False
                    time0=time.time()
                    min_schema, cost_increment = self.partition(collector, wLoad, cur_par_schema)
                    operator_cost=DiskIo.compute_repartitioning_cost(cur_par_schema,min_schema,wLoad.attrs_length)
                    self.optimize_time+=time.time()-time0
                    total_actions+=1
                    total_blocks += DiskIo.compute_cost(collector, cur_par_schema, wLoad.attrs_length)
                    if min_schema!=cur_par_schema and cost_increment>operator_cost:
                        true_actions+=1
                        self.action_list[sql['time']] = min_schema
                        total_rep_blocks+=operator_cost
                        cur_par_schema=min_schema
                    collector=np.array([])
            collector=np.append(collector,sql['feature'])
        if collector.shape[0]>0:
            total_blocks += DiskIo.compute_cost(collector, cur_par_schema, wLoad.attrs_length)
        self.action_ratio=[true_actions,round(true_actions/total_actions,3)]
        total_freq=(sum([sql['feature'].frequency for sql in wLoad.sql_list]))
        return total_blocks/total_freq,total_rep_blocks/total_freq