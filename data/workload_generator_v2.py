# two types of workload distribution
# a uniform distribution with 30, 000 distinct transactions 
# a Zipf distribution with 23, 457 distinct transactions (parameter s=1.16 , obey the 8-2 rule.)
# Poisson distribution =>how many transactions are submitted to the system during each interval
# vary the percentage of tuples α we need to repartition, which varies from 100% to 20%.

import random
import math
import pandas as pd
from pandas.core.frame import DataFrame
import numpy as np
import os
def maxminnorm(arr):
    maxrows = arr.max(axis=1)
    minrows = arr.min(axis=1)
    data_shape = arr.shape
    data_rows = data_shape[0]
    data_cols = data_shape[1]
    t = np.empty((data_rows, data_cols))
    for i in range(data_rows):
        t[i, :] = (arr[i, :] - minrows[i]) / (maxrows[i] - minrows[i])
    return t

def generate_random_query(attr_num_complete,query_num,total_time,start_time,maximum_sigma_percentage):
    # choose access attributes
    # 每个查询所有属性的方差波动范围都是一样的
    # 将属性范围随机切分成5份,随机选中某一份   [[0,10],[20,40],[5,35],[40,50],[30,49]]
    cut_num=5
    attr_split_list=[]
    min_length=round(attr_num_complete/cut_num)
    start_list=[random.randint(i*5,(i+1)*5-1) for i in range(10)]
    while(len(attr_split_list)<=5):
        # 随机确定长度
        split_length = random.randint(min_length, 3 * min_length)
        # 随机确定起始属性
        start_index = start_list[random.randint(0,4)]
        end_index=start_index+split_length-1
        if end_index<attr_num_complete:
            attr_split_list.append([start_index,end_index])
    # maximum_sigma_percentage=0.2
    # maximum_sigma_percentage=0.5
    querys=[]
    is_generate_perform=0
    access_attr_temp_list=[list() for _ in range(len(attr_split_list))]
    # uniform_num=int(query_num * 0.6)
    uniform_num=int(query_num)
    cnt=0
    while cnt<uniform_num:
    # for i in range(uniform_num):
        access_attr = list()
        range_integer = random.randint(0, len(attr_split_list) - 1)
        # 0~99
        range_lower, range_upper = attr_split_list[range_integer][0], attr_split_list[range_integer][1]
        # 分成三个点
        point_list=[range_lower+round(range_upper-range_lower)*th/4 for th in range(1,4)]
        selected_point = point_list[random.randint(0,2)]
        # 控制sql访问属性的数量
        access_attr_sum = random.randint(1, range_upper-range_lower+1)
        if access_attr_sum>6:access_attr_sum=6  #访问属性数量不超过10个
        for i in range(range_lower, range_upper + 1):
            sigma = random.uniform(0, (range_upper - selected_point) * maximum_sigma_percentage)
            attr_value = math.ceil(random.gauss(selected_point, sigma))
            if attr_value > range_upper:
                attr_value = range_upper
            elif attr_value < range_lower:
                attr_value = range_lower
            if attr_value in access_attr:
                continue
            else:
                access_attr.append(attr_value)
            if len(access_attr) == access_attr_sum: break
        rep_times=random.randint(1,5)
        if cnt+rep_times>uniform_num: rep_times=uniform_num-cnt
        [access_attr_temp_list[range_integer].append(access_attr) for _ in range(rep_times)]
        cnt+=rep_times
    access_attr_list=list()
    for item in access_attr_temp_list:access_attr_list+=item
    phase=10
    steps=int(total_time / (2 * phase))
    poisson_data=np.array([])
    base_index=start_time
    query_num_rest=query_num
    for i in range(steps):
        item_size=query_num_rest if i==steps-1 else int(query_num/steps)
        poisson_item=np.random.poisson(lam=phase, size=item_size)+base_index
        poisson_data=np.concatenate([poisson_data,poisson_item])
        base_index+=2*phase
        query_num_rest-=int(query_num/steps)
    poisson_data=poisson_data.astype(int)
    poisson_data=sorted(poisson_data)
    # 为access_attr_list 生成其他的特征
    for iter,access_attr in enumerate(access_attr_list):
        # choose frequency
        if is_generate_perform<=5:
            freq=random.randint(round(query_num/10),round(query_num/8))
            is_generate_perform+=1
        else:
            # 干扰query ：1~20之间的整数
            freq=random.randint(1,20)

        # choose access keys
        # access_attr范围内之间的a个整数,其中0<=a<=2
        scan_key_num=min(random.randint(0,2),len(access_attr))
        scan_attr=[]
        for i in random.sample(access_attr,scan_key_num):
            scan_attr.append(str(i))
        scan_attr=",".join(scan_attr)

        # choose selectivity
        selectivity=round(0.2*random.random(),3)
        time=poisson_data[iter]
        attr_list_str = [str(attr) for attr in access_attr]
        querys.append([",".join(attr_list_str), freq, scan_attr, selectivity, time])
    return querys


def main():
    # attr_num_list=[30,50,100,200]
    # attr_num_list=[30,50,60,65,75,80,100,125,150,175]
    # for i in range(1,6):
    #     for attr_num in attr_num_list:
    #         # attr_num=100
    #         query_num=1000
    #         querys=generate_random_query(attr_num,query_num)
    #         df=DataFrame(querys)
    #         print(querys)
    #         df.to_csv("/home/liupengju/pycharmProjects/SCVP-V3/data/random%d/%dattr.csv"%(i,attr_num),header=0,index=0)
    
    # query_num_list=[200,500,1000,2000,5000]
    # query_num_list=[300,500,500]  #产生3个时间段，每个时段300+500+500=1300数量查询的负载集，
    # T=[60,20,60] #负载流的时间（必须是20的倍数）
    maximum_sigma_percentage_list=[0.4]
    query_num_list=[1300]
    T=[400]

    # maximum_sigma_percentage_list=[0.3,0.4,0.5,0.6,0.7]
    # query_num_list=[400,400,400,400,400]
    # T=[100,300,400,200,200]
    attr_range = 50
    basedir=os.getcwd()
    for i in range(2,3):
        new_path=os.path.join(basedir, 'data' + str(i))
        querys=[]
        if not os.path.exists(new_path):
            os.mkdir(new_path)
        for idx,query_num in enumerate(query_num_list):
            # query_num=1000
            start_time=sum(T[:idx]) if idx>0 else 0
            # start_time=278
            querys+=generate_random_query(attr_range,query_num,T[idx],start_time,maximum_sigma_percentage_list[idx])
        sorted(querys)
        df=DataFrame(querys,columns=['access_attr','freq','scan_attr','selectivity','time'])
        df.sort_values(by='time',axis=0,ascending=True,inplace=True)
        # df.sort_values('time')
        df.to_csv(os.path.join(basedir,"data%d/%dquery-steam.csv"%(i,sum(query_num_list))),header=0,index=0)

if __name__=="__main__":
    main()


