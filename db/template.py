import psycopg2
import time
from utils import read_query_data
from utils import list_solved_list

tpch_tables = {
    'nation': {
        'attrs': ["N_NATIONKEY", "N_NAME", "N_REGIONKEY", "N_COMMENT"],
        'types': ["INTEGER", "CHAR(25)", "INTEGER", "VARCHAR(152)"]
    },
    'region': {
        'attrs': ["R_REGIONKEY", "R_NAME", "R_COMMENT"],
        'types': ["INTEGER", "CHAR(25)", "VARCHAR(152)"]
    },
    'part': {
        'attrs': ["P_PARTKEY", "P_NAME", "P_MFGR", "P_BRAND", "P_TYPE", "P_SIZE", "P_CONTAINER", "P_RETAILPRICE",
                  "P_COMMENT"],
        'types': ["INTEGER", "VARCHAR(55)", "CHAR(25)", "CHAR(10)", "VARCHAR(25)", "INTEGER", "CHAR(10)",
                  "DECIMAL(15,2)", "VARCHAR(23)"]
    },
    'supplier': {
        'attrs': ["S_SUPPKEY", "S_NAME", "S_ADDRESS", "S_NATIONKEY", "S_PHONE", "S_ACCTBAL", "S_COMMENT"],
        'types': ["INTEGER", "CHAR(25)", "VARCHAR(40)", "INTEGER", "CHAR(15)", "DECIMAL(15,2)", "VARCHAR(101)"]
    },
    'partsupp': {
        'attrs': ["PS_PARTKEY", "PS_SUPPKEY", "PS_AVAILQTY", "PS_SUPPLYCOST", "PS_COMMENT"],
        'types': ["INTEGER", "INTEGER", "INTEGER", "DECIMAL(15,2)", "VARCHAR(199)"]
    },
    'customer': {
        "attrs": ["C_CUSTKEY", "C_NAME", "C_ADDRESS", "C_NATIONKEY", "C_PHONE", "C_ACCTBAL", "C_MKTSEGMENT",
                  "C_COMMENT"],
        "types": ["INTEGER", "VARCHAR(25)", "VARCHAR(40)", "INTEGER", "CHAR(15)", "DECIMAL(15,2)", "CHAR(10)",
                  "VARCHAR(117)"]
    },
    'orders': {
        'attrs': ["O_ORDERKEY", "O_CUSTKEY", "O_ORDERSTATUS", "O_TOTALPRICE", "O_ORDERDATE", "O_ORDERPRIORITY",
                  "O_CLERK", "O_SHIPPRIORITY", "O_COMMENT"],
        'types': ["INTEGER", "INTEGER", "CHAR(1)", "DECIMAL(15,2)", "DATE", "CHAR(15)", "CHAR(15)", "INTEGER",
                  "VARCHAR(79)"]
    },
    'lineitem': {
        'attrs': ["L_ORDERKEY", "L_PARTKEY", "L_SUPPKEY", "L_LINENUMBER", "L_QUANTITY", "L_EXTENDEDPRICE", "L_DISCOUNT",
                  "L_TAX", "L_RETURNFLAG", "L_LINESTATUS", "L_SHIPDATE", "L_COMMITDATE", "L_RECEIPTDATE",
                  "L_SHIPINSTRUCT", "L_SHIPMODE", "L_COMMENT"],
        'types': ["INTEGER", "INTEGER", "INTEGER", "INTEGER", "DECIMAL(15,2)", "DECIMAL(15,2)", "DECIMAL(15,2)",
                  "DECIMAL(15,2)", "CHAR(1)", "CHAR(1)", "DATE", "DATE", "DATE", "CHAR(25)", "CHAR(10)", "VARCHAR(44)"]
    }
}


# def generate_subtable_by_partition(querys,tb_name,partitions):
def cal_subtable_by_partition_time(querys, path, partitions):
    tb_name = path.split("/")[-1].replace(".csv", "")
    cardinity = getCardinityByTable(tb_name)
    create_tb_sql = ""
    insert_tb_sql = ""
    drop_tb_sql = ""
    for index, partition in enumerate(partitions):
        sub_tab_name = tb_name + "sub" + str(index)
        create_tb_sql += "CREATE TABLE %s ( %s %s NOT NULL" % (
        sub_tab_name, tpch_tables[tb_name]['attrs'][0], tpch_tables[tb_name]['types'][0])
        for idx, attr_id in enumerate(partition):
            if (attr_id == 0):
                continue
            create_tb_sql += ",\n%s %s NOT NULL" % (
            tpch_tables[tb_name]['attrs'][attr_id], tpch_tables[tb_name]['types'][attr_id])
            # if idx==len(partition)-1:
        create_tb_sql += ");\n"
        # 插入数据语句
        if (partition.count(0) > 0):
            partition.remove(0)
        attr_list = ",".join([tpch_tables[tb_name]["attrs"][x] for x in [0] + partition])
        insert_tb_sql += "INSERT INTO %s (SELECT %s from %s);\n" % (sub_tab_name, attr_list, tb_name)
        drop_tb_sql += "DROP TABLE %s;\n" % (sub_tab_name)
    print(create_tb_sql)
    print(insert_tb_sql)
    print(drop_tb_sql)

    original_query = ""
    update_query = ""
    # affinity_matrix,querys=read_query_data(r"D:\pycharmProjects\SCVP\data\tpch\customer.csv",8)
    # partitions.append([])
    for query in querys:
        attr_list = ",".join(
            [tpch_tables[tb_name]["attrs"][attr_idx] for attr_idx, attr_val in enumerate(query['value']) if
             attr_val == 1])
        if (len(query['scan_key']) == 0):
            original_query += ("SELECT %s FROM %s;\n" % (attr_list, tb_name)) * query['freq']
        else:
            original_query += ("SELECT %s FROM %s LIMIT %d;\n" % (
            attr_list, tb_name, cardinity * query['selectivity'])) * query['freq']

    for query in querys:
        attr_indxs = [attr_idx for attr_idx, attr_val in enumerate(query['value']) if attr_val == 1]
        for idx, partition in enumerate(partitions):

            if (list_solved_list(attr_indxs, partition)):
                subtb_name = tb_name + "sub" + str(idx)
                # if(partition.count(0)==0):
                #     partition.append(0)
                attr_list = ",".join([tpch_tables[tb_name]["attrs"][attr] for attr in attr_indxs if attr in partition])
                if (len(query['scan_key']) == 0):
                    update_query += ("SELECT %s FROM %s;\n" % (attr_list, subtb_name)) * query['freq']
                else:
                    update_query += ("SELECT %s FROM %s LIMIT %d;\n" % (
                    attr_list, subtb_name, cardinity * query['selectivity'])) * query['freq']
    print(original_query)
    print(update_query)
    return vpExperiment(create_tb_sql, insert_tb_sql, drop_tb_sql, original_query, update_query)


def getPgConn():
    # 创建连接对象
    conn = psycopg2.connect(database="postgres", user="postgres", password="root", host="localhost", port="5432")
    return conn


def getCardinityByTable(tb_name):
    conn = getPgConn()
    cur = conn.cursor()
    cur.execute("SELECT * FROM %s;" % tb_name)
    # results=cur.fetchall()
    resnum = cur.rowcount
    cur.close()
    conn.close()
    return resnum


def vpExperiment(create_tb_sql, insert_tb_sql, drop_tb_sql, original_query, update_query):
    conn = getPgConn()
    cur = conn.cursor()  # 创建指针对象

    # # 创建表
    for line in create_tb_sql.split(";\n")[:-1]:
        cur.execute(line + ";")
    conn.commit()
    # 插入数据
    for line in insert_tb_sql.split("\n")[:-1]:
        cur.execute(line)
    conn.commit()
    time1 = time.time()
    # lines=original_query.split("\n")
    for line in original_query.split("\n")[:-1]:
        cur.execute(line)
        # print("%s 执行结果：affect %d",cur.rowcount)
    conn.commit()

    time2 = time.time()
    for line in update_query.split("\n")[:-1]:
        cur.execute(line)
        # print("%s 执行结果：affect %d",cur.rowcount)
    conn.commit()

    time3 = time.time()
    print("原查询时间:", time2 - time1)
    print("分区后查询时间:", time3 - time2)

    # 删除表
    for line in drop_tb_sql.split("\n")[:-1]:
        cur.execute(line)
    conn.commit()
    # 关闭连接
    cur.close()
    conn.close()
    return (time2 - time1 - time3 + time2)


if __name__ == '__main__':
    # print(generate_subtable_by_partition("customer",[[2, 7], [1], [4, 5], [6], [3], [0]]))
    print(cal_subtable_by_partition_time("customer", "data/tpch/customer.csv", [[0, 3], [6], [4, 5], [1, 2, 4, 7]]))