import math
from controller.par_algorithm.scvp import Scvp
from controller.db.disk_io import DiskIo
from controller.util import Util
from controller.par_algorithm.fp_growth_plus import  load_data,Fp_growth_plus
import numpy as np
from itertools import chain
import time
class Smopdc:
    """
    1. performance checking component.
    2. the database repartitioning component.
    """
    def __init__(self):
        self.a=1-0.95
        # self.za=0.95 #the function of confidence level α
        # self.fn=0.1  #the ratio threshold of number of queries that satisfies physical read ratio threshold of a query
        # self.ca=0.1     #percision
        self.candidated_par_schemas=[]

    def prune_a_in_b(self,a,bb):
        b=bb.copy()
        for item in a:
            if item in b: b.remove(item)
        return b
    def remove_duplicate(self,x):
        res=[]
        for x1 in x:
            if x1 not in res:
                res.append(x1)
        return res

    def get_all_schema(self,current_par:list,rest_closed_frequency_set):
        # print('left:',current_par,' right:',rest_closed_frequency_set)
        i=0
        for target_par in reversed(current_par):
            current_par_copy=current_par.copy()
            rest_closed_frequency_set_copy=rest_closed_frequency_set.copy()
            current_par_copy.remove(target_par)
            solved_sets=[]
            for seti in reversed(rest_closed_frequency_set_copy):
                if Util.list_solved_list(seti,target_par):
                    solved_sets.append(seti)
                    rest_closed_frequency_set_copy.remove(seti)
            if len(solved_sets)==0:
                i+=1
                if i==(len(current_par)):
                    if rest_closed_frequency_set_copy:
                        change_item=rest_closed_frequency_set_copy.pop(0)
                        current_par_copy.append(change_item)
                        current_par_copy.append(target_par)
                        self.get_all_schema(current_par_copy,rest_closed_frequency_set_copy)
                    else:
                        self.candidated_par_schemas.append(current_par_copy+[target_par]+rest_closed_frequency_set_copy)
                continue
            solved_sets.append(target_par)
            solved_sets=self.remove_duplicate(solved_sets)
            for seti in solved_sets:
                new_current_par = current_par_copy.copy()
                new_current_par.append(seti)
                for x in solved_sets:
                    if x==seti:continue
                    new_current_par.append(self.prune_a_in_b(seti,x))
                self.get_all_schema(new_current_par,rest_closed_frequency_set_copy)


    def partition(self,windowsql,wLoad,cur_par_schema):
        windowsql=windowsql.tolist()
        X, accessedAttr = Util.prune_affinity_matrix(wLoad.compute_affinity_matrix_by_sqls(windowsql))
        # 1. get the closed item sets(CIS)
        data_set, _ = load_data(accessedAttr, windowsql)
        min_support = 0
        L_GLOBAL, _ = Fp_growth_plus().generate_L(data_set, min_support)
        closed_frequency_set=[]
        for itemset in reversed(L_GLOBAL):
            for key in itemset:
                flag=True
                for freq_set in closed_frequency_set:
                    if Util.list_in_list(key,freq_set):
                        flag=False
                        break
                if flag: closed_frequency_set.append([k for k in key])
        #2.  filter the CIS
        avg_frequency=sum([sql.frequency for sql in windowsql])/len(windowsql)
        As=[]
        for sql in windowsql:
            if sql.frequency>=avg_frequency:As.append(Util.transferAttrPos(sql.attributes))
        for aas in As:
            for itemset in reversed(closed_frequency_set):
            # if sum([sql.frequency if Util.list_in_list(itemset,self.transferAttrPos(sql.attributes)) else 0 for sql in windowsql])<avg_frequency:
                if Util.list_in_list(itemset,aas) and len(itemset)<=len(aas):
                    closed_frequency_set.remove(itemset)
        new_closed_frequency_set=As+closed_frequency_set
        #3. generate the candidate vertical partitioning solution
        self.candidated_par_schemas=[]
        self.get_all_schema([new_closed_frequency_set[0]],new_closed_frequency_set[1:])
        min_cost=init_cost=DiskIo.compute_cost(windowsql,cur_par_schema,wLoad.attrs_length)
        min_schema=cur_par_schema
        for schema in self.candidated_par_schemas:
            merged_schema= list(chain.from_iterable(schema.copy()))
            for attr in range(len(wLoad.attrs_length)):
                if attr not in merged_schema:
                    schema.append([attr])
            cost=DiskIo.compute_cost(windowsql,schema,wLoad.attrs_length)
            if cost<min_cost:
                min_schema=schema
                min_cost=cost
        return min_schema,init_cost-min_cost

    def compute_z_score(self,wLoad):
        sel_list=[]
        for sql in wLoad.sql_list:
            for _ in range(sql['feature'].frequency):
                sel_list.append(sql['feature'].selectivity)
        sorted(sel_list,reverse=False)
        return sel_list[int(0.95*len(sel_list))]
    def repartition(self,initial_par,wLoad):
        threshod_f=0.06
        n=fn=0
        self.action_list = dict()
        for sql in wLoad.sql_list:
            if sql['feature'].selectivity>=threshod_f:fn+=sql['feature'].frequency
            n+=sql['feature'].frequency
        fn/=n
        self.za=self.compute_z_score(wLoad)
        self.ca=(2*self.za-0.2)/2 #confidence interval
        N=math.ceil((self.za**2)*(fn*(1-fn)/(self.ca**2)))
        print(N)
        collector=np.array([])
        cur_par_schema=initial_par
        total_blocks=0
        total_rep_blocks=0
        self.optimize_time=0
        total_actions = 0
        true_actions = 0
        for sql in wLoad.sql_list:
            collector=np.append(collector,sql['feature'])
            if collector.shape[0]>=N:
                time0 = time.time()
                min_schema,cost_increment=self.partition(collector,wLoad,cur_par_schema)
                operator_cost=DiskIo.compute_repartitioning_cost(cur_par_schema,min_schema,wLoad.attrs_length)
                self.optimize_time+=time.time()-time0
                total_blocks += DiskIo.compute_cost(collector, cur_par_schema, wLoad.attrs_length)
                total_actions+=1
                if min_schema!=cur_par_schema and cost_increment>operator_cost:
                    true_actions+=1
                    self.action_list[sql['time']] = min_schema
                    total_rep_blocks+=operator_cost
                    cur_par_schema=min_schema
                collector=np.array([])
        if collector.shape[0]>0:
            total_blocks += DiskIo.compute_cost(collector, cur_par_schema, wLoad.attrs_length)
        self.action_ratio = [true_actions, round(true_actions / total_actions, 3)]
        total_freq=(sum([sql['feature'].frequency for sql in wLoad.sql_list]))
        return total_blocks/total_freq,total_rep_blocks/total_freq