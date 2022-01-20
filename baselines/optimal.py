from controller.baselines.after_all import Afterall
from controller.db.workload import Workload
from controller.environment.env_opt import EnvOpt
import numpy as np
class OptimalController(Afterall):
    def repartition(self, initial_par, wLoad, **kwargs):
        self.action_list=dict()
        total_actions = 0
        good_actions = 0
        env = EnvOpt(wLoad)
        action_map = np.ones(2000, dtype=int)
        for action in action_map:
            obs, reward, done, info=env.step(action)
            if reward!=0:total_actions+=1
            if reward>0:good_actions+=1
            if env.done:break
        self.action_list=env.action_list
        self.optimize_time=env.optimize_time
        self.action_ratio = [len(self.action_list.keys()), round(good_actions/total_actions,3)]
        total_freq=sum([sql['feature'].frequency for sql in env.w.sql_list])
        total_cost=sum(env.io_cost) / total_freq
        print(f"Average query cost: {total_cost}")
        return total_cost,sum(env.rep_cost)/total_freq

if __name__=="__main__":
    oc=OptimalController()
    wLoad = Workload(50, '../data/data1/5000query-steam.csv')
    # wLoad.prune_sql_list(1500, 2001)
    oc.repartition(None,wLoad)
    print(oc.action_list.keys())