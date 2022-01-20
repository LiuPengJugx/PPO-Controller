from controller.db.workload import Workload
from stable_base_ppo import PpoController
from baselines.optimal import OptimalController
from baselines.feedback import Feedback
from db.transaction import Transaction
from db.par_management import ParManagement as PM
from controller.db.disk_io import DiskIo
from controller.baselines.smopdc import Smopdc
from controller.baselines.after_all import Afterall
from controller.baselines.apply_all import Applyall
from controller.baselines.feedback import Feedback
from controller.baselines.dyvep import Dyvep
from controller.baselines.no_controller import NoController
from controller.baselines.optimal import OptimalController
from db.jta import JTA
from util import Util
import numpy as np
np.set_printoptions(threshold=np.inf)
class AdapterController:
    # algorithms = [Afterall(),Smopdc(),Feedback(),PpoController(),OptimalController()]
    # Specify algorithm
    algorithms = [Applyall(),Smopdc()]
    def experiment(self,file_path):
        res_map=[]
        self.wLoad = Workload(50, file_path)
        initial_par = [[i for i in range(50)]]
        for algo in self.algorithms:
            algo.repartition(initial_par,self.wLoad)
            res_map.append(self.get_schema_metrics(algo.action_list))
        return res_map

    def compute_affinity_matrix_change(self,action_list:dict):
        start_time=0
        last_aff_mat=self.wLoad.compute_affinity_matrix_consider_selectivity(0,0)
        last_par=[[i for i in range(50)]]
        # aff_mat_change=dict()
        for time in action_list.keys():
            if len(action_list[time])>0:
                print('-'*100)
                print(f"Time Range:[{start_time},{time}]")
            cur_aff_mat=self.wLoad.compute_affinity_matrix_consider_selectivity(start_time,time+1)
            aff_mat_change,accessed_att=Util.prune_affinity_matrix(last_aff_mat - cur_aff_mat)
            if len(action_list[time]) > 0:
                print(last_par)
                print(f"Partition Schema: {[par for par in last_par if Util.list_solved_list(accessed_att,par)]};\n Time: {time} ;\n Matrix Change: \n{aff_mat_change,accessed_att}")
            last_aff_mat=cur_aff_mat.copy()
            if len(action_list[time]) > 0:
                last_par = action_list[time]
                print(last_par)
                print(f"After Partition: {[par for par in last_par if Util.list_solved_list(accessed_att, par)]}")
            start_time=time+1


    def get_schema_metrics(self,action_list):
        txn_list=self.process_workload()
        metrics=dict()
        txn_count=0
        exec_scale,latency,rep_latency=0.0,0.0,0.0
        jta=JTA()
        latency_list=list()
        throughput_list=list()
        # cur_par_schema = [[i for i in range(50)]]
        for key in txn_list.keys():
            queries=txn_list[key]
            t=Transaction(queries,jta,key)
            result=t.call()
            txn_count+=result.txn_count
            latency+=result.costTime
            latency_list.append(round(result.costTime,3))
            if key in action_list.keys():
                if action_list[key]:
                    result2=t.repartition(action_list[key])
                    # cur_par_schema=action_list[key]
                    rep_latency+=result2.finishTime-result2.startTime
                    # throughput_list.append(round(txn_count / ((result.costTime+rep_latency*0.5) / 60),3))
                    throughput_list.append([txn_count,round(result.costTime,3),round(rep_latency,3)])
            else:
                # throughput_list.append(round(txn_count / (result.costTime / 60),3))
                throughput_list.append([txn_count, round(result.costTime,3), 0])

        metrics['Latency']=latency
        metrics['Rep_latency']=rep_latency
        metrics['Throughput']=txn_count/latency
        metrics['latency_list']=latency_list
        metrics['throughput_list']=throughput_list
        self.clear_table_data(jta)
        print(f'Action Length:{len(action_list)} , Content:{action_list.keys()}', )
        print(latency_list)
        return metrics

    def clear_table_data(self,jta):
        drop_tb_sql=""
        for index in PM.cur_partitions.keys():
            sub_tab_name = PM.test_tables['name'] + "sub" + str(index)
            drop_tb_sql += "DROP TABLE %s;\n" % (sub_tab_name)
        for line in drop_tb_sql.split("\n")[:-1]:
            jta.query(line)
        jta.commit()
        jta.close()
        PM.cur_partitions=dict()

    def process_workload(self)->dict:
        # w=Workload(Conf.TABLE_ATTRIBUTE_NUM, file_path)
        txn_list=dict()
        cur_time=0
        queries=[]
        for idx,item in enumerate(self.wLoad.sql_list):
            if item['time']!=cur_time:
                if len(queries)>0: txn_list[self.wLoad.sql_list[idx-1]['time']]=queries.copy()
                cur_time=item['time']
                queries.clear()
            queries.append(item['feature'])
        txn_list[self.wLoad.sql_list[-1]['time']] = queries.copy()
        return txn_list

if __name__=='__main__':
    ac=AdapterController()
    result=dict()
    # Specify dataset
    workload_dict = {
        'data1': [3000, 1300, 4000],
        # 'data1': [1500, 3000, 1300, 4000],
        'data2': [1200, 1350],
        'data3': [1600, 2600]
    }
    for path in workload_dict.keys():
        for query_num in workload_dict[path]:
            result[query_num]=ac.experiment(f'data/{path}/{query_num}query-steam.csv')
    print(result)