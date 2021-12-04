import pymysql
conn = pymysql.connect(host='localhost', user='root', password='chunjay606', db='data_science')
curs = conn.cursor(pymysql.cursors.DictCursor)

sql = """select sno, midterm, final 
        from score 
        where midterm >= 20 and final >= 20
        order by sno"""
curs.execute(sql)

row = curs.fetchone()
while row:
    print(row)
    row = curs.fetchone()

curs.close()
conn.close()