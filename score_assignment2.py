# mysql 테이블 생성 및 데이터 추가
import pandas as pd
import pymysql

xl_file = '/Users/JaehoByun/JB/_School/2021_2 데이터사이언스/과제및시험/score.xlsx'
df = pd.read_excel(xl_file)

conn = pymysql.connect(host='localhost', user='root', password='chunjay606', db='data_science')
curs = conn.cursor(pymysql.cursors.DictCursor)

# 테이블 생성
mk_table_sql = """create table if not exists score
    (sno int primary key,
    attendance float,
    homework float,
    discussion int,
    midterm float,
    final float,
    score float,
    grade varchar(3))"""
curs.execute(mk_table_sql)

# 데이터 넣기
insert_sql = """insert into score(sno, attendance, homework, discussion, midterm, final, score, grade)
                values (%s, %s, %s, %s, %s, %s, %s, %s)"""
for idx in range(len(df)):
    curs.execute(insert_sql, tuple(df.values[idx]))
conn.commit()

# 데이터 삽입 확인
show_table_sql = 'select * from score'
curs.execute(show_table_sql)

row = curs.fetchone()
while row:
    print(row)
    row = curs.fetchone()

curs.close()
conn.close()