import time

from db.jta import JTA
"""
测试HDD-I成本模型与PG数据库实际环境的差异
"""
my_jta=JTA()
test_times=10


# time0=time.time()
# for _ in range(test_times):
#     my_jta.query("SELECT * FROM tt_tab WHERE a2='1' and a3='5' and a4='1020'")
# print('Example 1:',(time.time()-time0)/10)
#
#
# time1=time.time()
# for _ in range(test_times):
#     my_jta.query("SELECT * FROM tt_tab WHERE a2<>'1' and a3<>'5' and a4<>'1020'")
# print('Example 2:',(time.time()-time1)/10)
#
# time2=time.time()
# for _ in range(test_times):
#     my_jta.query("SELECT * FROM tt_tab WHERE a2<>'1'")
# print('Example 3:',(time.time()-time2)/10)
#
#
# time3=time.time()
# for _ in range(test_times):
#     my_jta.query("SELECT * FROM tt_tab")
# print('Example 4:',(time.time()-time3)/10)
#
#
# time4=time.time()
# for _ in range(test_times):
#     my_jta.query("SELECT a1,a2 FROM tt_tab")
# print('Example 5:',(time.time()-time4)/10)

# Example 1: 0.013916373252868652
# Example 2: 0.5094051122665405
# Example 3: 0.5083346128463745
# Example 4: 0.4503688097000122
# Example 5: 0.035331439971923825

# 重复执行多次后
# Example 1: 0.016120219230651857
# Example 2: 0.3979427576065063
# Example 3: 0.40204148292541503
# Example 4: 0.396588659286499
# Example 5: 0.03543694019317627


# 结论
# 1：执行时间 好像与取出数据的量级有关系，如 Example3 与 4，虽然3有WHERE子句，但于4执行时间差不多。Example 1明显小于4，因为1中判定条件无任何行结果。
# Example5明显小于4，因为5中只取出两行结果。
# Example 1与Example 5时间接近，但由于Example 5 取出两列数据，仍然大于Example 1 执行时间。

# 2：WHERE添加只会增加少部分执行时间，并不起绝对作用


# 3: 重复执行多次后，Example2 /3 /4 执行时间稍微缩短，说明WHERE子句 以及 全表扫描 具有缓存功能？



# 拆分成多表，进行测试
# my_jta.query("DROP TABLE IF EXISTS tt_tab_sub;\n CREATE TABLE tt_tab_sub (a0 SERIAL,a2 CHAR(4),a3 CHAR(4))")
# my_jta.query("INSERT INTO tt_tab_sub (SELECT a0,a2,a3 FROM tt_tab)")
# my_jta.commit()
# time5=time.time()
# for _ in range(test_times):
#     my_jta.query("SELECT a0,a2,a3 FROM tt_tab WHERE a2<>'1' and a3<>'5'")
# print('Example 6:',(time.time()-time5)/10)
#
#
# time6=time.time()
# for _ in range(test_times):
#     my_jta.query("SELECT a2,a3 FROM tt_tab_sub")
# print('Example 7:',(time.time()-time6)/10)
#
# time7=time.time()
# for _ in range(test_times):
#     my_jta.query("SELECT * FROM tt_tab_sub WHERE a2<>'1' and a3<>'5'")
# print('Example 8:',(time.time()-time7)/10)
#
# time8=time.time()
# for _ in range(test_times):
#     my_jta.query("SELECT * FROM tt_tab_sub WHERE a0='1' and a2='1' and a3='5'")
# print('Example 9:',(time.time()-time8)/10)


# Example 6: 0.04241459369659424
# Example 7: 0.024752044677734376
# Example 8: 0.033130741119384764
# Example 9: 0.006550335884094238

"""
结论：
1： 同等情况下，小表相对于大表，当查询数据量不变，时间基本不变，大表时间略长一点点。 如Example 5与 Example 7 （0.035略大于0.024）， 
Example 6与 Example 8（0.042略大于0.033）， Example 1与 Example 9（0.0139略大于0.0077）
"""

time9=time.time()
for _ in range(test_times):
    my_jta.query("SELECT * FROM tt_tab_sub limit 10000")
print('Example 10:',(time.time()-time9)/10)

time10=time.time()
for _ in range(test_times):
    my_jta.query("SELECT * FROM tt_tab_sub limit 10")
print('Example 11:',(time.time()-time10)/10)

time11=time.time()
for _ in range(test_times):
    my_jta.query("SELECT * FROM tt_tab_sub")
print('Example 12:',(time.time()-time11)/10)

# Example 10: 0.01700148582458496
# Example 11: 0.0014611482620239258
# Example 12: 0.07124898433685303

"""
结论：
1.同等条件下，limit子句涉及的元组数量越少，查询时间越短
"""
