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
workload_dict={
    # 'data1':[1500,3000,1300,4000],
    'data1':[1500],
    # 'data2':[1200,1350],
    # 'data3':[1600,2600]
}
# for query_num in [1500,3000,10000,12000,2600,4000,5000,1300]:
res_per_workload=dict()
for path in workload_dict.keys():
    for query_num in workload_dict[path]:
        wLoad=Workload(50,f'data/{path}/{query_num}query-steam.csv')
        # wLoad.prune_sql_list(1500,2001)
        # wLoad.prune_sql_list(0,1500)
        initial_par=[[i for i in range(50)]]
        # Average query cost: [475.10063994204296, 275.1814779038879, 275.6893262496981, 266.0294011108428, 263.0802342429365, 483.3342791596233]
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
            # result.append(algo.optimize_time)
            # result.append(algo.action_ratio)
        print(result)
        res_per_workload[query_num]=result
print(res_per_workload)

# 1500query-steam.csv
# Optimal Afterall  Applyall  Smopdc   Feedback  Piggyback   Dyvep     DQN       PPO     DDPG     Double-DDPG    Hierarchical-RL   No-Controller
# 221.51   475.1    275.18   275.681   266.02    263.085     483.3             252.61                                                   2957.848

# 2600query-steam.csv
# Optimal Afterall  Applyall  Smopdc   Feedback  Piggyback   Dyvep     DQN       PPO     DDPG     Double-DDPG    Hierarchical-RL   No-Controller
# 289.15   436.76    391.66   388.15    362.94    1415.85                                                                              2836.88

# 3000query-steam.csv
# Optimal Afterall  Applyall  Smopdc   Feedback  Piggyback   Dyvep     DQN       PPO    DDPG   Double-DDPG    Hierarchical-RL   No-Controller   A2C
# 218.01   485.09    275.14    277.60    259.49     256.63    495.46              243                                             2947.39     405.57


# 4000query-steam.csv
# Optimal Afterall  Applyall  Smopdc   Feedback  Piggyback   Dyvep     DQN       PPO    DDPG   Double-DDPG    Hierarchical-RL  No-Controller
# 318.14    688.59   471.66   464.68    446.93                970.92              441                                               2906.81

# 5000query-steam.csv
# Optimal Afterall  Applyall  Smopdc   Feedback  Piggyback   Dyvep     DQN       PPO    DDPG   Double-DDPG    Hierarchical-RL  No-Controller
# 325.09  763.94      480.16  467.53   447.92                 877.24                                                             2922.77


# 10000query-steam.csv
# Optimal  Afterall  Applyall  Smopdc   Feedback  Piggyback   Dyvep     DQN     PPO    DDPG   Double-DDPG    Hierarchical-RL
# 250.98  442.26     322.12    314.18     300.76              490.52           282.83


# 12000query-steam.csv
# Optimal Afterall  Applyall  Smopdc   Feedback  Piggyback   Dyvep     DQN       PPO    DDPG   Double-DDPG    Hierarchical-RL  No-Controller
# 312.94  669.64     465.13   461.94    453.31               726.34            446.15                                            2973.29


#  1500: [[20, 0.833], [451, 0.43], [320, 0.64], [231, 0.411], [149, 0.603]],
#  3000: [[21, 0.955], [848, 0.394], [768, 0.512], [262, 0.289], [160, 0.656]],
#  10000: [[85, 0.885], [4113, 0.699], [2533, 0.76], [1299, 0.833], [768, 0.751]],
#  12000: [[95, 1.0], [5524, 0.853], [4284, 0.714], [1394, 0.952], [1251, 0.813]]}
#  2600: [[13, 1.0], [1139, 0.78], [1431, 0.55], [201, 0.935], [158, 0.843]],
#  4000: [[23, 1.0], [1852, 0.863], [523, 0.916], [367, 0.976], [311, 0.831]],
#  5000: [[20, 1.0], [2306, 0.856], [1341, 0.805], [380, 0.941], [340, 0.836]],
# {1300: [[4, 1.0], [599, 0.856], [474, 0.729], [104, 0.945], [96, 0.867]],