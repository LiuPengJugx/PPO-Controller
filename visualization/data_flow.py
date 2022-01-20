import matplotlib.pyplot as plt
from controller.db.workload import Workload
plt.rcParams['figure.figsize']=(12.0,8.0)
def show_workload_flow(wLoad:Workload):
    # time_list=[time for time in  range(wLoad.sql_list[0]['time'],wLoad.sql_list[-1]['time'])]
    start_time=0
    # end_time=wLoad.sql_list[-1]['time']
    end_time=50
    last_time=0
    cnt=0
    for sql in wLoad.sql_list:
        cur_time=sql['time']
        if start_time <= cur_time <= end_time:
            query=sql['feature']
            if cur_time!=last_time:
                last_time=cur_time
                cnt=0
            else:
                cnt+=0.1
                cnt+=0
            solved_attrs = [i for i, x in enumerate(query.attributes) if x == 1]
            plt.plot([cur_time*1+cnt]*len(solved_attrs),solved_attrs,'-o')
    # plt.show()
    plt.savefig(f'{len(wLoad.sql_list)}/{start_time}-{end_time}.svg',bbox_inches='tight',pad_inches=0)



if __name__=='__main__':
    for query_num in [1300]:
        wLoad=Workload(50,f'../data/data1/{query_num}query-steam-copy.csv')
        show_workload_flow(wLoad)

