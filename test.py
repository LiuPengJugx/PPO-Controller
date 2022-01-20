import numpy as np
from controller.db.workload import Workload
from controller.environment.env import Env
from controller.environment.multenv import Multenv
from controller.environment.multenv_v2 import Multenv2
from controller.environment.env_v2 import Env2
from controller.environment.env_v3 import Env3
from controller.environment.env_v2_state import EnvState
from controller.environment.env_v2_2600 import EnvSkew4000
from optimal_controller import get_args,watch
from optimal_controller_ppo import run_ppo
def test_workload():
    wLoad=Workload(50)
    wLoad.load_sql('data/5000query.csv')
    aff_matrix=wLoad.compute_affinity_matrix(0,13)
    print(aff_matrix)
    print(wLoad.sql_list)

def test_multenv():
    env=Multenv()
    env.seed(20)
    # env.reset()
    a=env.step([493,496])
    b=env.step([5,3])
    c=env.step([10,8])
    d=env.step([40,30])
    e=env.step([50,60])
    f=env.step([300,250])
    print(a)
    print(b)
    print(c)
    print(d)
    print(e)
    print(f)

def test_multenv2():
    env=Multenv2()
    env.seed(20)
    # env.reset()
    env.step([20,11])
    env.step([47,23])
    env.step([37,17])
    env.step([48,47])
    env.step([32,20])
    env.step([39,30])
    env.step([31,43])
    env.step([16,47])
    env.step([12,4])
    env.step([24,16]) # 290-306 matrix


def test_env():
    env=Env()
    env.seed(1)
    action_list=[0, 6, 14, 22, 30, 38, 46, 54, 62, 70, 78, 86, 94, 102, 110, 118, 126, 134, 142, 150, 158, 166, 202, 238, 274,310, 346, 382, 418, 454,500]
    for idx,action in enumerate(action_list[1:]):
        env.step(action-action_list[idx])
    print('io_cost : ', env.io_cost)
    print(sum(env.io_cost)/sum([sql['feature'].frequency for sql in  env.w.sql_list]))

    #250.4519439748853
def test_env2():
    env = Env2()
    action_list=np.zeros(498,dtype=int)
    # action_list[4]=1
    # action_list[6]=1
    # action_list[200]=1
    # action_list[300]=1
    # action_list[400]=1
    # action_list[450]=1
    for action in action_list:
        env.step(action)
    print('Length:', len(env.io_cost), ' ; io_cost : ', env.io_cost)
    print(f"Average query cost: {sum(env.io_cost) / sum([sql['feature'].frequency for sql in env.w.sql_list])}")
    # Length: 2;
    # io_cost: [273082.0, 8017566.0]
    # Average query cost: 245.9767986945557
def test_env_state(wLoad):
    # env = EnvState(wLoad)
    env = EnvSkew4000(wLoad)
    action_list=np.zeros(498,dtype=int)
    action_list[4]=1
    action_list[6]=1
    action_list[10]=1
    action_list[30]=1
    action_list[40]=1
    action_list[45]=1
    for action in action_list:
        env.step(action)
    print('Length:', len(env.io_cost), ' ; io_cost : ', env.io_cost)
    print(f"Average query cost: {sum(env.io_cost) / sum([sql['feature'].frequency for sql in env.w.sql_list])}")

def test_env3():
    env = Env3()
    action_list=np.ones(500,dtype=int)
    # action_list[2]=1
    # action_list[200]=1
    # action_list[300]=1
    # action_list[400]=1
    # action_list[450]=1
    for action in action_list:
        env.step(action)
    print('Length:', len(env.io_cost), ' ; io_cost : ', env.io_cost)
    print(f"Average query cost: {sum(env.io_cost) / sum([sql['feature'].frequency for sql in env.w.sql_list])}")
    # 1500query-steam
    # Average query cost: (+Operator Cost) 233.3251630041053

    # 3000query-steam
    # Average query cost: 224.21901794985908 (+Operator Cost 229.43889630618602) (Optimal)

    # 10000query-steam (time range: 1500-2000)
    # Average query cost: 265.31126099269636
def test():
    import numpy as np
    from sklearn.cluster import SpectralClustering
    X = np.loadtxt("error_data.txt")
    for k in range(1, len(X) + 1):
        try:
            y_pred = SpectralClustering(n_clusters=k, affinity='precomputed', random_state=0).fit_predict(X)
        except ValueError as e:
            print(111111111111111)
            continue
        print(y_pred)
def test_algorithm(args=get_args()):
    # run_ql(args)
    run_ppo()
def watch_agent(args=get_args()):
    watch(args)
if __name__ == '__main__':
    # test_workload()
    # test_env()
    # test_env2()
    # test_env3()
    # test_multenv()
    # test_multenv2()
    # test()
    # test_algorithm()
    # watch_agent()

    wLoad = Workload(50, 'data/data1/2600query-steam.csv')
    test_env_state(wLoad)