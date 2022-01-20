import pandas as pd
import numpy as np
from controller.db.sql import Sql
from controller.db.conf import Conf
from controller.db.disk_io import DiskIo
from controller.util import Util
class Workload:
    def __init__(self,attr_num,filepath):
        self.attr_num = attr_num
        self.attrs_length = [4 for _ in range(self.attr_num)]
        self.sql_list=np.array([])
        self.load_sql(filepath)

    def compute_affinity_matrix(self,start,end):
        n=self.attr_num
        affinity_matrix=np.array(([[0]*n])*n)
        for item in self.sql_list:
            if start <= item['time'] < end:
                sql=item['feature']
                value=[0]*n
                for column_i,val in enumerate(sql.attributes):
                    if val==1:
                        value[column_i]=1
                        affinity_matrix[column_i][column_i]+=sql.frequency
                        # 更新亲和度矩阵
                        for i,v in enumerate(value[:column_i]):
                            if v==1:
                                affinity_matrix[i][column_i]+=sql.frequency
                                affinity_matrix[column_i][i]+=sql.frequency
        return affinity_matrix

    def update_affinity_matrix(self,last_affinity_matrix,start):
        n=self.attr_num
        affinity_matrix=last_affinity_matrix
        for item in self.sql_list:
            if item['time'] == start:
                sql=item['feature']
                value=[0]*n
                for column_i,val in enumerate(sql.attributes):
                    if val==1:
                        value[column_i]=1
                        affinity_matrix[column_i][column_i]+=sql.frequency
                        # 更新亲和度矩阵
                        for i,v in enumerate(value[:column_i]):
                            if v==1:
                                affinity_matrix[i][column_i]+=sql.frequency
                                affinity_matrix[column_i][i]+=sql.frequency
            elif item['time'] > start:
                break
        return affinity_matrix


    def compute_affinity_matrix_consider_selectivity(self,start,end):
        n=self.attr_num
        affinity_matrix=np.array(([[0]*n])*n)
        for item in self.sql_list:
            if start <= item['time'] < end:
                sql=item['feature']
                value=[0]*n
                for column_i,val in enumerate(sql.attributes):
                    if val==1:
                        value[column_i]=1
                        affinity_matrix[column_i][column_i]+=sql.frequency*sql.selectivity*100
                        # 更新亲和度矩阵
                        for i,v in enumerate(value[:column_i]):
                            if v==1:
                                affinity_matrix[i][column_i]+=sql.frequency*sql.selectivity*100
                                affinity_matrix[column_i][i]+=sql.frequency*sql.selectivity*100
        return affinity_matrix

    def update_affinity_matrix_consider_selectivity(self,last_affinity_sel_matrix,start):
        n=self.attr_num
        affinity_matrix=last_affinity_sel_matrix.copy()
        for item in self.sql_list:
            if item['time'] == start:
                sql=item['feature']
                value=[0]*n
                for column_i,val in enumerate(sql.attributes):
                    if val==1:
                        value[column_i]=1
                        affinity_matrix[column_i][column_i]+=sql.frequency*sql.selectivity*100
                        # 更新亲和度矩阵
                        for i,v in enumerate(value[:column_i]):
                            if v==1:
                                affinity_matrix[i][column_i]+=sql.frequency*sql.selectivity*100
                                affinity_matrix[column_i][i]+=sql.frequency*sql.selectivity*100
            elif item['time'] > start:
                break
        return affinity_matrix

    def compute_affinity_matrix_by_sqls(self,sqls):
        n=self.attr_num
        affinity_matrix=np.array(([[0]*n])*n)
        for sql in sqls:
            value=[0]*n
            for column_i,val in enumerate(sql.attributes):
                if val==1:
                    value[column_i]=1
                    affinity_matrix[column_i][column_i]+=sql.frequency
                    # 更新亲和度矩阵
                    for i,v in enumerate(value[:column_i]):
                        if v==1:
                            affinity_matrix[i][column_i]+=sql.frequency
                            affinity_matrix[column_i][i]+=sql.frequency
        return affinity_matrix

    def transfer_sql_vector(self, sqls):
        sql_vector = np.zeros(self.attr_num,dtype=float)
        for sql in sqls:
            value=np.zeros(self.attr_num,dtype=float)
            for column_i, val in enumerate(sql.attributes):
                if val==1:
                    value[column_i]=1
            value*=sql.frequency*sql.selectivity
            sql_vector+=value
        return sql_vector

    def load_sql_by_time_range(self,start,end):
        filter_result=[]
        for item in self.sql_list:
            if start<= item['time'] < end:
                filter_result.append(item['feature'])
            elif item['time']>=end:
                break
        return filter_result

    def transfer_partition_distribution_feature_vector(self,last_par_schema,sqls,average_cost,last_par_feature_vector):
        feature_vector = np.zeros(self.attr_num)
        total_feq = 0
        if sqls:
            sql_weights=np.array([DiskIo.compute_cost([sql], last_par_schema, self.attrs_length)/sql.frequency for sql in sqls])
            sql_weights=sql_weights/average_cost
            for idx,sql in enumerate(sqls):
                columns=[]
                for column_i, val in enumerate(sql.attributes):
                    if val==1:
                        columns.append(column_i)
                accessed_par_num=sum([1 for par in last_par_schema if Util.list_solved_list(par,columns)])
                feature_vector[accessed_par_num-1]+=sql_weights[idx]*sql.frequency
                total_feq+=sql.frequency
        avg_feature_vector = feature_vector / total_feq if total_feq > 0 else feature_vector
        return avg_feature_vector

    def transfer_partition_distribution_feature_vector2(self,last_par_schema,sqls,last_par_feature_vector):
        feature_vector = np.zeros(self.attr_num)
        if sqls:
            cur_avg_cost=DiskIo.compute_cost(sqls,last_par_schema,self.attrs_length)/sum([sql.frequency for sql in sqls])
            feature_vector[0]=cur_avg_cost
        if last_par_feature_vector[0] > 0:
            feature_vector[0]=(last_par_feature_vector[0]-feature_vector[0])/last_par_feature_vector[0]
        return feature_vector


    def mask_affinity_matrix(self,cur_matrix,par_schema):
        matrix=cur_matrix.copy()
        if len(par_schema)>1:
            for par in par_schema:
                for attr1 in par:
                    for attr2 in par:
                        # if attr1==attr2:continue
                        matrix[attr1][attr2]=0
        return matrix

    def mask_affinity_matrix_improvement(self,cur_mask_affinity_matrix,time,par_schema):
        mask_affinity_matrix=cur_mask_affinity_matrix.copy()
        for item in self.sql_list:
            if time <= item['time'] < time+1:
                sql = item['feature']
                weight=sql.frequency * sql.selectivity * 100
                solved_attrs = [i for i, x in enumerate(sql.attributes) if x == 1]
                for attr1 in solved_attrs:
                    for attr2 in solved_attrs:
                        if attr1==attr2:continue
                        solved_pars=[par for par in par_schema if attr1 in par and attr2 in par]
                        if attr1!=attr2 and len(solved_pars)==0:
                            mask_affinity_matrix[attr1][attr2] += weight
                for par in par_schema:
                    if Util.list_solved_list(solved_attrs,par):
                        par_rest_attrs=list(set(par)-set(solved_attrs))
                        for attr1 in par_rest_attrs:
                            for attr2 in par_rest_attrs:
                                if attr1!=attr2:
                                    mask_affinity_matrix[attr1][attr2] += weight*(len(par_rest_attrs)/len(par))
        return mask_affinity_matrix


    def mask_par_matrix(self,sqls,par_schema):
        n = self.attr_num
        affinity_matrix = np.array(([[0] * n]) * n)
        # 对1个分区的情况进行特殊处理
        if len(par_schema)==1:
            par_schema=[[i] for i in range(n)]
        for sql in sqls:
            accessed_attrs=[]
            for column_i, val in enumerate(sql.attributes):
                if val == 1:
                    accessed_attrs.append(column_i)
            attr_group=[[] for _ in par_schema]
            for attr in accessed_attrs:
                for idx,par in enumerate(par_schema):
                    if attr in par:
                        attr_group[idx].append(attr)
                        break
            attr_group_width=[]
            attr_groups=[]
            for idx,attr_g in enumerate(attr_group):
                if attr_g:
                    attr_groups.append(attr_g)
                    attr_group_width.append(sum([self.attrs_length[attr] for attr in par_schema[idx]]))

            for idx1,attr_g1 in enumerate(attr_groups):
                for idx2,attr_g2 in enumerate(attr_groups):
                    if attr_g1==attr_g2:continue
                    total_freq=sql.frequency * sql.selectivity *100*attr_group_width[idx2]
                    for attr1 in attr_g1:
                        for attr2 in attr_g2:
                            affinity_matrix[attr1][attr2]= total_freq/(len(attr_g1)*len(attr_g2))
        return affinity_matrix

    def load_sql(self,filepath):
        df=pd.read_csv(filepath,header=None)
        for row in range(df.shape[0]):
            # parse attribute
            attrs=[0]*self.attr_num
            for attr in df.iloc[row][0].split(','):
                attrs[int(attr)]=1
            # ________________________________________end
            # parse scan keys of sql statement 
            scan_keys=[]
            if (df.iloc[row][2])!=(df.iloc[row][2]): # if math.isnan(df.iloc[row][2]):
                pass
            else:
                scan_keys=[int(x) for x in df.iloc[row][2].split(",")]
            # _________________________________________end

            self.sql_list=np.append(self.sql_list,{
                'time':df.iloc[row][4],
                'feature':Sql(attrs,df.iloc[row][3],scan_keys,df.iloc[row][1])
            })

    def prune_sql_list(self,begin,end):
        new_sql_list=np.array([])
        for sql in self.sql_list:
            if begin<=sql['time']<end:
                new_sql_list=np.append(new_sql_list,sql)
        self.sql_list=new_sql_list