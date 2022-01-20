from controller.baselines.after_all import Afterall
from controller.util import Util
import numpy as np
from controller.db.disk_io import DiskIo
class Dyvep(Afterall):
    qt = []
    def statistic_collector(self,collector):
        # queries table QT  [id frequency description execution-info ]
        # attribute_usage_table(AUT)
        # clustered_affinity_table CAT
        # attribute affinity table AAT
        record_num=0
        for sql in collector:
            is_first=True
            access_attributes=Util.transferAttrPos(sql.attributes)
            for qt_item in self.qt:
                if qt_item['attributes']==Util.transferAttrPos(sql.attributes):
                    qt_item['frequency']+=1
                    record_num+=1
                    is_first=False
                    break
            if is_first:
                self.qt.append({'id':len(self.qt)+1,'attributes':access_attributes,'frequency':1,'description':''})
        if record_num==len(collector):
            return True
        return False

    def repartition(self, initial_par, wLoad):
        cycle=8
        collector = np.array([])
        iter=0
        start_time=wLoad.sql_list[0]['time']
        isFirst = True
        total_blocks=0
        total_rep_blocks = 0
        self.action_list = dict()
        cur_par_schema = initial_par
        for idx,sql in enumerate(wLoad.sql_list):
            if sql['time']>=(iter*cycle) and sql['time']<((iter+1)*cycle):
                collector = np.append(collector, sql['feature'])
            elif sql['time']>=((iter+1)*cycle) or idx==len(wLoad.sql_list)-1:
                total_blocks += DiskIo.compute_cost(collector, cur_par_schema, wLoad.attrs_length)
                if self.statistic_collector(collector) or isFirst:
                    isFirst=False
                    self.qt=[]
                    min_schema, cost_increment=self.partition(collector, wLoad, cur_par_schema)
                    operator_cost = DiskIo.compute_repartitioning_cost(cur_par_schema, min_schema,wLoad.attrs_length)
                    if min_schema != cur_par_schema and cost_increment>operator_cost:
                        self.action_list[sql['time']] = min_schema
                        total_rep_blocks += operator_cost
                        cur_par_schema=min_schema
                collector = np.array([sql['feature']])
                iter+=1
        if collector.shape[0] > 0:
            total_blocks += DiskIo.compute_cost(collector, cur_par_schema, wLoad.attrs_length)
        total_freq = (sum([sql['feature'].frequency for sql in wLoad.sql_list]))
        return total_blocks/total_freq,total_rep_blocks/total_freq