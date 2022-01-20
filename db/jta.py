import random

from controller.db.pg import Pg
from controller.db.par_management import ParManagement as PM

class JTA:
    def __init__(self):
        self.conn=Pg().get_conn()
        self.cur = self.conn.cursor()
    # def begin(self):
    #     self.conn.begin()

    def commit(self):
        self.conn.commit()

    def rollback(self):
        self.conn.rollback()

    def cancel(self):
        self.conn.cancel()

    def query(self, sql):
        self.cur.execute(sql)

    def close(self):
        self.cur.close()
        self.conn.close()

# 创建初始表
if __name__=='__main__':
    jta=JTA()

    create_statement=f"DROP TABLE IF EXISTS tt_tab;\nCREATE TABLE tt_tab ( {PM.test_tables['attrs'][0]} {PM.test_tables['types'][0]} PRIMARY KEY"
    for idx in range(1,50):
        create_statement+=f",\n{PM.test_tables['attrs'][idx]} {PM.test_tables['types'][idx]} NOT NULL"
    create_statement += ");\n"
    print(create_statement)
    jta.query(create_statement)
    jta.commit()
    cardinality = 100000
    for cnt in range(1,cardinality+1):
        insert_statement=f"INSERT INTO tt_tab({','.join([PM.test_tables['attrs'][i] for i in range(1,50)])}) VALUES ({','.join([str(random.randint(1000,9999)) for _ in range(49)])})"
        jta.query(insert_statement)
    jta.commit()
    jta.close()