from controller.baselines.after_all import Afterall
import numpy as np
from controller.db.disk_io import DiskIo
class NoController(Afterall):
    def repartition(self, initial_par, wLoad, **kwargs):
        self.action_list=dict()
        collector = np.array([])
        for idx, sql in enumerate(wLoad.sql_list):
            collector = np.append(collector, sql['feature'])
        return DiskIo.compute_cost(collector, initial_par, wLoad.attrs_length)/sum([sql['feature'].frequency for sql in wLoad.sql_list])


if __name__=="__main__":
    nc=NoController()
    nc.repartition(None,None)
    print(nc.action_list)