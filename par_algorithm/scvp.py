from controller.par_algorithm.ColumnCluster_v2 import ColumnCluster
class Scvp:
    @staticmethod
    def partitioner(affinity_matrix,temp_workload,attrs_length):
        schema=ColumnCluster().compute_cost_by_spectal_cluster(affinity_matrix,temp_workload,attrs_length)
        return schema

    def partitioner2(affinity_matrix,temp_workload,attrs_length):
        schema=ColumnCluster().compute_cost_by_spectal_cluster2(affinity_matrix,temp_workload,attrs_length)
        return schema