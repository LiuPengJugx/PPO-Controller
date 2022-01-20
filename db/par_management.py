class ParManagement:
    test_tables = {
        'name':"tt_tab",
        'attrs': ["a" + str(i) for i in range(50)],
        'types': ["SERIAL"]+["CHAR(4)" for _ in range(49)]
    }
    # cur_partitions=[[i+1 for i in range(50)]]
    cur_partitions=dict()