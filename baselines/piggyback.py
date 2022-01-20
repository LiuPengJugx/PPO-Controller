from controller.baselines.after_all import Afterall
from controller.db.disk_io import DiskIo
import numpy as np
from controller.par_algorithm.pid import PID
from controller.util import Util
class Piggyback(Afterall):
    def repartition(self, initial_par, wLoad):
        SP = 254.5  # SP:the desired output value    /    PV:the actual measured value
        repartition_threshold = -20
        pid = PID(1.2, 1, 0.001)
        pid.SetPoint = SP  # 评价每个查询的执行成本
        pid.setSampleTime(1)
        collector = {'content': np.array([]),
                     'start': 0}
        cur_par_schema = initial_par
        total_blocks = 0
        total_sql_num = 0
        last_solved_schema=[]
        for sql in wLoad.sql_list:
            collector['content'] = np.append(collector['content'], sql['feature'])
            if collector['content'].shape[0] == 1:
                collector['start'] = sql['time']
                pid.setLastTime(sql['time'])
            time_interval = sql['time'] - collector['start'] + 1
            if time_interval == 1: continue
            PV = (DiskIo.compute_cost(collector['content'], cur_par_schema, wLoad.attrs_length) + total_blocks) \
                 / ((total_sql_num + sum([item.frequency for item in collector['content']])))
            pid.update(PV, sql['time'])
            print('PV:', PV, '  output:', pid.output)
            if pid.output <= repartition_threshold:
                min_schema, cost_increment = self.partition(collector['content'], wLoad, cur_par_schema)

                if min_schema != cur_par_schema:
                    operator_cost = DiskIo.compute_repartitioning_cost(cur_par_schema, min_schema, wLoad.attrs_length)
                    # 记录哪些分区被分区操作所影响
                    cur_par_schema_copy=cur_par_schema.copy()
                    cur_solved_schema=[]
                    for query in collector['content']:
                        pos=Util.transferAttrPos(query.attributes)
                        # piggyback normal transactions into repartition transaction
                        for par in last_solved_schema:
                            if Util.list_solved_list(pos, par):
                                operator_cost-=50
                                break
                        for par in cur_par_schema_copy:
                            if Util.list_solved_list(pos,par):
                                cur_solved_schema.append(par)
                                cur_par_schema_copy.remove(par)
                    last_solved_schema=cur_solved_schema
                    # ~~~~~~~~~~end
                    total_blocks += DiskIo.compute_cost(collector['content'], cur_par_schema, wLoad.attrs_length)
                    total_blocks += operator_cost
                    total_sql_num += sum([item.frequency for item in collector['content']])
                    print("时刻", sql['time'], ",更新分区方案为:", min_schema, ",预计成本收益为:", cost_increment)
                    cur_par_schema = min_schema
                    collector['content'] = np.array([])
        # 279.8375392417291
        if collector['content'].shape[0] > 0:
            total_blocks += DiskIo.compute_cost(collector['content'], cur_par_schema, wLoad.attrs_length)
        return total_blocks / (sum([sql['feature'].frequency for sql in wLoad.sql_list]))