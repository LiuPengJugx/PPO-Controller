import math
from controller.par_algorithm.pid import PID
from controller.db.disk_io import DiskIo
from controller.baselines.optimal import OptimalController
import numpy as np
from controller.baselines.after_all import Afterall
from scipy import optimize
from sympy import *
from autograd import grad
import time

from controller.util import Util


class Feedback(Afterall):
    # def measure_feedback_value(self,temp_collector,time_interval,wLoad,cur_par_schema):
    #     sql_count = sum([item.frequency for item in temp_collector)
    #     PV = DiskIo.compute_cost(collector['content'][:idx + 1], cur_par_schema, wLoad.attrs_length) / (
    #                 (sql['time'] -) * sql_count)

    def ziegler_nichols(self):
        # G(x)=Ke^(-tx)/(Tx+1)   其中K为比例系数；T为惯性时间常数；τ为纯延迟时间常数
        K, T, t = 2, 30, 10
        t=np.arange(0,300,0.02)
        def fun(x):
            y = K * (math.e ** (-t * x)) / (T * x + 1)
            return y
        def my_grad():
            x=symbols('x')
            y=K*x/(T*x+1)
            return diff(y,x)
        fun_grad=my_grad()
        def fun2(x):
            y = -K / (x**2+0.4 * x + 1)
            return y
        fun_grad2=grad(fun2)

        # print(fun_grad.evalf(subs={'x':3}))

        # step(fun)
        # t1,c1=step2(system=lti([K],[T,1]),T=t)
        # plt.plot(t,[fun_grad2(x) for x in t],'r',label='s1 Step Response',linewidth=0.5)
        # plt.show()
        # maximum=optimize.fminbound(lambda x:-fun_grad.evalf(subs={'x':x}),0,300)
        maxpoint=optimize.fminbound(lambda x:-fun_grad2(x),0,300) #切点横坐标
        maxy=fun2(maxpoint)# 切点纵坐标
        maxr=fun_grad2(maxpoint) #斜率
        # 切线与y轴交点(即常数项)
        tangent_b=maxy-maxr*maxpoint
        # 切线与x轴交点
        tt=(0-tangent_b)/maxr
        TT=maxpoint-tt
    #   更新PID参数
        kp=1.2*TT/tt
        ki=2*tt
        kd=0.5*tt
        return kp,ki,kd

    def repartition(self, initial_par, wLoad, **kwargs):
        # SP如何设定？？？？？SP应该是期望的平均成本
        # optimalController=OptimalController()
        # SP=optimalController.repartition(initial_par,wLoad) # SP:the desired output value    /    PV:the actual measured value
        # SP=289.1506 # SP:the desired output value    /    PV:the actual measured value
        # repartition_threshold=-30
        repartition_threshold=0
        isFirst = True
        start_time=wLoad.sql_list[0]['time']
        collector = {'content':[],
                     'start':start_time}
        pid = PID(1, 0.001, 0)
        # pid = PID(1, 0, 0)
        # pid.SetPoint=SP #评价每个查询的执行成本
        pid.setSampleTime(0)
        pid.setLastTime(start_time)
        cur_par_schema = initial_par
        self.action_list =dict()
        total_blocks=[]
        total_rep_blocks = 0
        total_sql_num=0
        self.optimize_time=0
        total_actions = 0
        true_actions = 0
        last_time=0
        cost_time_map={"cost":list(),"cdf":list()}
        for cur_time in range(wLoad.sql_list[0]['time'],wLoad.sql_list[-1]['time']+1):
        # for sql in wLoad.sql_list:
            collector['content'] += wLoad.load_sql_by_time_range(cur_time,cur_time+1)
            PV=(DiskIo.compute_cost(collector['content'], cur_par_schema, wLoad.attrs_length)+sum(total_blocks))\
               /((total_sql_num+sum([item.frequency for item in collector['content']])))
            if pid.SetPoint==0:
                pid.SetPoint = PV-10
            pid.update(PV,cur_time)
            print('PV:', PV, '  output:', pid.output)
            if pid.output>=100:
                pid.SetPoint = PV - 10
            if pid.output>=5:
                pid.SetPoint-=10
            """
            启发式的调参方法：如果PID的误差总是低于阈值，且最近10次，调节的效果不好，output始终维持在某个值附近，说明此时环境已经达到稳定状态，不适合调节，阈值过高，因此需要重新设置阈值为OUTPUT均值-5，是PID停止调节，随着环境的
            改变，当前状态不适合，OUTPUT值降低，低于阈值，此时，PID需要重新调节，因此可以更新阈值为OUTPUT均值+10。
            """
            # if len(latest_output_set) >= 10:
            #     latest_output_set.pop(0)
            # latest_output_set.append(pid.output)
            # print('标准差：', np.std(latest_output_set, ddof=1))  # 求标准差

            # if len(latest_output_set) >= 10 and np.std(latest_output_set, ddof=1) < 0.5 and is_add:
            #     repartition_threshold = np.mean(latest_output_set) - 5
            #     latest_output_set.clear()
            #     is_add=False
            if pid.output<=repartition_threshold or isFirst:
                isFirst=False
                # if len(latest_output_set) >= 10 and np.std(latest_output_set, ddof=1) < 0.5 and not is_add:
                #     repartition_threshold = np.mean(latest_output_set) + 10
                #     latest_output_set.clear()
                #     is_add = True
                time0 = time.time()
                min_schema, cost_increment = self.partition(collector['content'], wLoad, cur_par_schema)
                # temp_cur_par_schema = cur_par_schema.copy()
                # if len(temp_cur_par_schema) == 1 and len(temp_cur_par_schema[0]) == 50:
                #     temp_cur_par_schema = [[num] for num in range(50)]
                # # new par需要加上之前未替换的分区
                # for par in temp_cur_par_schema:
                #     temp_par = par.copy()
                #     for par_new in min_schema:
                #         if Util.list_solved_list(par_new, temp_par):
                #             [temp_par.remove(attr) for attr in par_new if attr in temp_par]
                #     if temp_par: min_schema.append(temp_par)
                operator_cost = DiskIo.compute_repartitioning_cost(cur_par_schema, min_schema, wLoad.attrs_length)
                self.optimize_time+=time.time()-time0
                total_actions+=1
                if min_schema!=cur_par_schema and cost_increment>operator_cost:
                    true_actions+=1
                    self.action_list[cur_time]=min_schema
                    cost_blocks=DiskIo.compute_cost(collector['content'], cur_par_schema, wLoad.attrs_length)
                    total_blocks.append(cost_blocks)
                    cost_time_map['cost'].append(f"({last_time+1} {cur_time})->{cost_blocks}")
                    cost_time_map['cdf'].append(f"{cur_time}->{sum(total_blocks)}")
                    last_time=cur_time
                    total_rep_blocks +=operator_cost
                    total_sql_num+=sum([item.frequency for item in collector['content']])
                    print("时刻",cur_time,",与上一时刻间隔负载成本为:",cost_blocks,",更新分区方案为:",min_schema)
                    cur_par_schema=min_schema
                    collector['content']=[]
        # 282.7867
        if len(collector['content']) > 0:
            total_blocks.append(DiskIo.compute_cost(collector['content'], cur_par_schema, wLoad.attrs_length))
        self.action_ratio = [true_actions, round(true_actions / total_actions, 3)]
        total_freq=(sum([sql['feature'].frequency for sql in wLoad.sql_list]))
        print(cost_time_map['cost'])
        print(cost_time_map['cdf'])
        return sum(total_blocks)/total_freq , total_rep_blocks/total_freq


# [9187.0, 15376.0, 6814.0, 24602.0, 24646.0, 28276.0, 7631.0, 31946.0, 26608.0, 11712.0, 20599.0, 3504.0, 10477.0]
# [9187.0, 15376.0, 6814.0, 24602.0, 24670.0, 35586.0, 29601.0, 87683.0]
