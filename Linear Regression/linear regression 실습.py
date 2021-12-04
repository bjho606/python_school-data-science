import pymysql
import numpy as np
import statsmodels.api as sm

conn = pymysql.connect(host='localhost', user='root', 
                         password='chunjay606', db='data_science')
curs = conn.cursor(pymysql.cursors.DictCursor)

sql = "select * from score"
curs.execute(sql)
data = curs.fetchall()

curs.close()
conn.close()

#, t['midterm'], t['final']
X = [ (t['attendance'], t['homework'], t['discussion'], t['midterm'])  for t in data ]
X = np.array(X)
X = sm.add_constant(X)

y = [ (t['score']) for t in data ]
y = np.array(y)

model = sm.OLS(y, X)
results = model.fit()
print(results.summary())

