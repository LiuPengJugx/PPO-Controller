from controller.db.workload import Workload
from controller.baselines.smopdc import Smopdc
from controller.baselines.after_all import Afterall
from controller.baselines.apply_all import Applyall
from controller.baselines.feedback import Feedback
from controller.baselines.dyvep import Dyvep
from controller.baselines.random_dynamic import RandomDynamic
from controller.baselines.no_controller import NoController
from controller.baselines.optimal import OptimalController
import os
basePath=os.getcwd()
# Specify datasets
workload_dict={
    'data1':[1500,3000,1300,4000],
    'data2':[1200,1350],
    'data3':[1600,2600]
}
res_per_workload=dict()
for path in workload_dict.keys():
    for query_num in workload_dict[path]:
        wLoad=Workload(50,f'data/{path}/{query_num}query-steam.csv')
        initial_par=[[i for i in range(50)]]
        # Specify algorithm
        # algorithms=[NoController(),RandomDynamic(),Afterall(),Applyall(),Smopdc(),Feedback(),Dyvep(),OptimalController()]
        algorithms=[Smopdc()]
        result=[]
        # cnt=0
        # for sql in wLoad.sql_list:
        #     cnt+=sql['feature'].frequency
        # result.append(wLoad.sql_list[-1]['time'])
        for algo in algorithms:
            avg_accessed_cost,avg_operate_cost=algo.repartition(initial_par, wLoad)
            result.append([round(avg_accessed_cost,2),round(avg_operate_cost,2),algo.action_ratio])
        print(result)
        res_per_workload[query_num]=result
print(res_per_workload)