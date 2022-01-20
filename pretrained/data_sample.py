import random
import numpy as np
import pandas as pd
from controller.db.conf import Conf
from controller.db.workload import Workload
from controller.baselines.optimal import OptimalController
class DataLoader:
    def __init__(self,file_path):
        self.w=Workload(Conf.TABLE_ATTRIBUTE_NUM,file_path)

    def compute_vector_change(self,cur_sqls,last_sqls):
        return self.w.transfer_sql_vector(cur_sqls) - self.w.transfer_sql_vector(last_sqls)

    def collect_par_vector_sample_optimal_controller(self,data_save_path):
        optimal_controller=OptimalController()
        average_cost=optimal_controller.repartition(None,self.w)
        start_time=self.w.sql_list[0]['time']
        data=[]
        action_list=optimal_controller.action_list
        restart_point=[key for key in action_list.keys() if not action_list[key]]
        [action_list.pop(key) for key in restart_point]
        action_keys=list(action_list.keys())
        # last_feature_vector=np.zeros(self.w.attr_num)
        # v3
        # last_feature_vector=self.w.compute_affinity_matrix_consider_selectivity(0, 0)

        for idx,end_time in enumerate(action_keys):
            # cur_par_schema=action_list[end_time]

            # v3
            # 由于预知性，先前推移1步
            end_time+=1

            if idx>0:
                cnt = idx - 1
                while not action_list[action_keys[cnt]]:
                    cnt -= 1
                last_par_schema = action_list[action_keys[cnt]]
            else: last_par_schema=[[i for i in range(self.w.attr_num)]]
            for time in range(start_time,end_time+1):
                # if time in restart_point: start_time=time
                sqls = self.w.load_sql_by_time_range(start_time, time + 1)
                # v1
                # cur_feature_vector=self.w.transfer_partition_distribution_feature_vector(last_par_schema,sqls,average_cost,last_feature_vector)
                # v2
                # cur_feature_vector=self.w.transfer_partition_distribution_feature_vector2(last_par_schema,sqls,last_feature_vector)
                # v3
                # temp_affinity_sel_matrix=self.w.compute_affinity_matrix_consider_selectivity(start_time, time + 1)
                # cur_feature_vector=self.w.mask_affinity_matrix(temp_affinity_sel_matrix,last_par_schema)
                # v4:   v3的版本取平均值
                cur_feature_vector=self.w.mask_par_matrix(sqls,last_par_schema)

                if time==end_time:
                    data.append(([1], cur_feature_vector.tolist()))
                    # v1
                    # last_feature_vector=self.w.transfer_partition_distribution_feature_vector(last_par_schema,sqls,average_cost,np.zeros(self.w.attr_num))
                    # v2
                    # last_feature_vector=self.w.transfer_partition_distribution_feature_vector2(last_par_schema,sqls,np.zeros(self.w.attr_num))
                    # v3
                    # last_feature_vector=temp_affinity_sel_matrix
                else:
                    data.append([[0],cur_feature_vector.tolist()])
            start_time=end_time+1
        pd.DataFrame(data).to_csv(data_save_path, header=0, index=0)

    # def collect_sample_from_attr_change(self):
    #     data=[]
    #     epoch=5
    #     while epoch>0:
    #         action_list=[]
    #         start_time = self.w.sql_list[0]['time']
    #         rand_end=random.randint(2,100)
    #         while start_time<self.w.sql_list[-1]['time']:
    #             start_time+=random.randint(1,rand_end)
    #             action_list.append(start_time)
    #         sqls,last_sqls=list(),list()
    #         last_attr_vector_change=None
    #         cnt=0
    #         for sql_dict in self.w.sql_list:
    #             if sql_dict['time']<=action_list[cnt]:
    #                 sqls.append(sql_dict['feature'])
    #             else:
    #                 cnt+=1
    #                 attr_vector_change=self.compute_vector_change(sqls,last_sqls)
    #                 if last_attr_vector_change is not None:
    #                     data.append((last_attr_vector_change.tolist(),attr_vector_change.tolist()))
    #                 last_attr_vector_change=attr_vector_change
    #                 last_sqls=sqls.copy()
    #                 sqls=list()
    #         attr_vector_change = self.compute_vector_change(sqls, last_sqls)
    #         data.append((last_attr_vector_change.tolist(), attr_vector_change.tolist()))
    #         epoch-=1
    #     pd.DataFrame(data).to_csv("attr_change_sample_12000.csv",header=0,index=0)

    def collect_sample_from_next_attr(self):
        data=[]
        epoch=5
        # epoch=1000
        while epoch>0:
            print("epoch: ",epoch)
            start_time = self.w.sql_list[0]['time']
            action_list=[start_time]
            end_time=self.w.sql_list[-1]['time']
            rand_end=random.randint(2,100)
            while start_time<end_time:
                start_time+=random.randint(1,rand_end)
                action_list.append(start_time)
            sqls=list()
            cnt=1
            for sql_dict in self.w.sql_list:
                if sql_dict['time']>action_list[cnt]:
                    cur_attr_vector=self.w.transfer_sql_vector(sqls)
                    next_time_end=action_list[cnt]+action_list[cnt]-action_list[cnt-1]
                    next_sqls=self.w.load_sql_by_time_range(action_list[cnt],next_time_end+1)
                    next_attr_vector=self.w.transfer_sql_vector(next_sqls)
                    if next_time_end<=end_time:
                        data.append((cur_attr_vector.tolist(),next_attr_vector.tolist()))
                    else:
                        break
                    cnt += 1
                    sqls=list()
if __name__=="__main__":
    # for i in [1300,2600,1500,3000,4000,5000]:
    for i in [1300,2600,5000]:
        dataLoader=DataLoader(f'../data/data1/{i}query-steam.csv')
        # dataLoader.collect_sample_from_next_attr()
        data_save_path=f"par_vector_sample_v4_{i}.csv"
        dataLoader.collect_par_vector_sample_optimal_controller(data_save_path)

