import numpy as np
from controller.baselines.after_all import Afterall
from controller.db.disk_io import DiskIo
import time
class Applyall(Afterall):
    def repartition(self, initial_par, wLoad):
        total_blocks=0
        total_rep_blocks=0
        collector=[]
        self.action_list = dict()
        cur_par_schema = initial_par
        self.optimize_time=0
        total_actions = 0
        true_actions = 0
        for cur_time in range(wLoad.sql_list[0]['time'], wLoad.sql_list[-1]['time'] + 1):
        # for idx,sql in enumerate(wLoad.sql_list):
            collector+=wLoad.load_sql_by_time_range(cur_time,cur_time+1)
            # if len(collector)>=8:
            time0 = time.time()
            min_schema, cost_increment = self.partition(collector, wLoad, cur_par_schema)
            operator_cost = DiskIo.compute_repartitioning_cost(cur_par_schema, min_schema, wLoad.attrs_length)
            self.optimize_time+=time.time()-time0
            total_actions+=1
            if min_schema!=cur_par_schema and cost_increment>operator_cost:
                true_actions+=1
                self.action_list[cur_time] = min_schema
                total_blocks+=DiskIo.compute_cost(collector,cur_par_schema,wLoad.attrs_length)
                total_rep_blocks+=operator_cost
                print("时刻",cur_time,",更新分区方案为:",min_schema,",预计成本收益为:",cost_increment)
                cur_par_schema=min_schema
                collector.clear()
        if len(collector)>0:
            total_blocks += DiskIo.compute_cost(collector, cur_par_schema, wLoad.attrs_length)
        # 254.56037189084762  最优情况下,平均每个查询的成本(不考虑分区操作成本)
        # 275.1814779038879(考虑分区操作成本)
        self.action_ratio = [true_actions, round(true_actions / total_actions, 3)]
        total_freq=(sum([sql['feature'].frequency for sql in wLoad.sql_list]))
        return total_blocks/total_freq,total_rep_blocks/total_freq
