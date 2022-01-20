from controller.db.workload import Workload
from controller.util import Util
from controller.par_algorithm.scvp import Scvp
import numpy as np
def partition_by_wide_workload(workload,wLoad):
    # 触发重分区
    attr_clusters = []
    # 根据最近的时间点查找 按时间连接的属性簇
    time_sequence = sorted(workload.keys(), reverse=True)
    cur_time=time_sequence[0]
    complete_attr_clusters = []
    cnt = 0
    while True:
        queries = workload[time_sequence[cnt]].copy()
        not_complete_attr_clusters = []
        # query的顺序会影响到cluster的分配，后续修改。
        # 即一个query没有任何一个可分配的cluster时，将其加入队列，等待其他query分配后，可能就有cluster可分配了！！！！！
        QUERIES=[]
        while len(queries)>0:
            item=queries.pop(0)
            QUERY={'Q': [item], 'A': [i for i, x in enumerate(item.attributes) if x == 1]}
            have_flag=True
            while have_flag:
                have_flag=False
                for query in queries.copy():
                    solved_attrs = [i for i, x in enumerate(query.attributes) if x == 1]
                    if Util.list_solved_list(solved_attrs,QUERY['A']):
                        QUERY['Q'].append(query)
                        [QUERY['A'].append(attr) for attr in solved_attrs if attr not in QUERY['A']]
                        queries.remove(query)
                        have_flag=True
            QUERIES.append(QUERY)
        for QUERY in QUERIES:
            solved_attrs =QUERY['A']
            is_exist = False
            will_combined_cluster = []
            for attr_cluster in attr_clusters:
                if Util.list_solved_list(attr_cluster['range'], solved_attrs):
                    # 该query遇到的第一个相关cluster
                    if not is_exist:
                        is_exist = True
                        attr_cluster['queries']+=QUERY['Q']
                        [attr_cluster['range'].append(attr) for attr in solved_attrs if attr not in attr_cluster['range']]
                    # 只要和query相关，都将该cluster添加到combine cluster队列中
                    will_combined_cluster.append(attr_cluster)
            # 合并combine cluster队列中的所有元素
            if will_combined_cluster:
                new_cluster = will_combined_cluster[0]
                if len(will_combined_cluster) > 1:
                    for item in will_combined_cluster[1:]:
                        new_cluster['queries'] += item['queries']
                        [new_cluster['range'].append(attr) for attr in item['range'] if
                         attr not in new_cluster['range']]
                    # 从attr_clusters中清除combine cluster队列中的所有元素
                    [attr_clusters.remove(item) for item in will_combined_cluster[1:]]
                    # attr_clusters.append(new_cluster)
                # 对有属性范围重合的属性簇进行合并
                if new_cluster not in not_complete_attr_clusters:
                    not_complete_attr_clusters.append(new_cluster)
            # 如果在第一、二轮负载中，未找到该查询相关的簇，因此需要为其分配新簇
            if not is_exist and cnt <= 1:
                cluster = {'queries': QUERY['Q'], 'range': solved_attrs}
                attr_clusters.append(cluster)
                not_complete_attr_clusters.append(cluster)


        # for query in queries:
        #     solved_attrs = [i for i, x in enumerate(query.attributes) if x == 1]
        #     is_exist = False
        #     will_combined_cluster = []
        #     for attr_cluster in attr_clusters:
        #         if Util.list_solved_list(attr_cluster['range'], solved_attrs):
        #             # 该query遇到的第一个相关cluster
        #             if not is_exist:
        #                 is_exist = True
        #                 attr_cluster['queries'].append(query)
        #                 [attr_cluster['range'].append(attr) for attr in solved_attrs if attr not in attr_cluster['range']]
        #             # 只要和query相关，都将该cluster添加到combine cluster队列中
        #             will_combined_cluster.append(attr_cluster)
        #     # 合并combine cluster队列中的所有元素
        #     if will_combined_cluster:
        #         new_cluster = will_combined_cluster[0]
        #         if len(will_combined_cluster) > 1:
        #             for item in will_combined_cluster[1:]:
        #                 new_cluster['queries'] += item['queries']
        #                 [new_cluster['range'].append(attr) for attr in item['range'] if
        #                  attr not in new_cluster['range']]
        #             # 从attr_clusters中清除combine cluster队列中的所有元素
        #             [attr_clusters.remove(item) for item in will_combined_cluster[1:]]
        #             # attr_clusters.append(new_cluster)
        #         # 对有属性范围重合的属性簇进行合并
        #         if new_cluster not in not_complete_attr_clusters:
        #             not_complete_attr_clusters.append(new_cluster)
        #     # 如果在第一、二轮负载中，未找到该查询相关的簇，因此需要为其分配新簇
        #     if not is_exist and cnt <= 1:
        #         cluster = {'queries': [query], 'range': solved_attrs}
        #         attr_clusters.append(cluster)
        #         not_complete_attr_clusters.append(cluster)

        # 清除已完成的属性簇
        for attr_cluster in attr_clusters.copy():
            if attr_cluster not in not_complete_attr_clusters:
                # 只要加入complete_attr_clusters列表中，就要判断是否具有重合的簇
                flag_combined = False
                for complete_cluster in complete_attr_clusters:
                    if Util.list_solved_list(complete_cluster['range'], attr_cluster['range']):
                        # 合并
                        complete_cluster['queries'] += attr_cluster['queries']
                        [complete_cluster['range'].append(attr) for attr in attr_cluster['range'] if
                         attr not in complete_cluster['range']]
                        flag_combined = True
                        break
                if not flag_combined: complete_attr_clusters.append(attr_cluster)
                attr_clusters.remove(attr_cluster)
        # 所有的属性簇在本轮负载中均未添加新的元素
        if not attr_clusters:
            break
        if cnt >= len(time_sequence)-1:
            # 时间达到尽头，若attr_clusters中有元素，应加入到complete_attr_clusters中
            # 但在加入前，要先判断是否有重合属性的cluster，需要进行合并
            for attr_cluster in attr_clusters:
                flag_combined = False
                for complete_cluster in complete_attr_clusters:
                    if Util.list_solved_list(complete_cluster['range'], attr_cluster['range']):
                        # 合并
                        complete_cluster['queries'] += attr_cluster['queries']
                        [complete_cluster['range'].append(attr) for attr in attr_cluster['range'] if
                         attr not in complete_cluster['range']]
                        flag_combined = True
                        break
                if not flag_combined: complete_attr_clusters.append(attr_cluster)
            break
        cnt += 1


    # 根据属性簇调用分区算法进行分区
    new_par_schema = []
    # 对不满足查询类别数的cluster进行剔除
    [complete_attr_clusters.remove(cluster) for cluster in complete_attr_clusters.copy() if len(cluster['queries'])<=2]
    for cluster in complete_attr_clusters:
        affinity_matrix = wLoad.compute_affinity_matrix_by_sqls(cluster['queries'])
        new_par_schema += Scvp.partitioner2(affinity_matrix, cluster['queries'], wLoad.attrs_length)
    print(f"回溯时间:{cur_time}--->{time_sequence[cnt]},聚簇数量:{len(complete_attr_clusters)},分区方案:{new_par_schema}")
    return new_par_schema,time_sequence[cnt]

def partitioner_simulation(wLoad):
    # collect_time=[9,15,19,27,33,50,120,170,188,200,230]
    collect_time = [88]
    workload=dict()
    for cur_time in range(wLoad.sql_list[0]['time'],wLoad.sql_list[-1]['time']+1):
        workload[cur_time]=wLoad.load_sql_by_time_range(cur_time,cur_time+1)
        # if cur_time in collect_time:
        #     partition_by_wide_workload(workload)
        partition_by_wide_workload(workload,wLoad)

def state_matrix(wLoad):
    cur_par=[[23,24,25],[2,3,4,5,6]]
    mask_affinity_matrix = np.array(([[0] * wLoad.attr_num]) * wLoad.attr_num)
    for cur_time in range(wLoad.sql_list[0]['time'], wLoad.sql_list[-1]['time'] + 1):
        mask_affinity_matrix=wLoad.mask_affinity_matrix_improvement(mask_affinity_matrix,cur_time,cur_par)

if __name__=='__main__':
    for query_num in [2000]:
        wLoad=Workload(50,f'../data/data2/{query_num}query-steam.csv')
        partitioner_simulation(wLoad)
        # state_matrix(wLoad)