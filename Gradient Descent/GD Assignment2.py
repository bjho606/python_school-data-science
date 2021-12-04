import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pymysql
import time
from  mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler

def load_dbscore_data():
    conn = pymysql.connect(host='localhost', user='root', 
                         password='chunjay606', db='data_science')
    curs = conn.cursor(pymysql.cursors.DictCursor)
    
    sql = "select * from score"
    curs.execute(sql)
    
    data  = curs.fetchall()
    
    curs.close()
    conn.close()
    
    X = [ ( t['attendance'], t['homework'], t['final'] ) for t in data ]
    X = np.array(X)
    
    y = [ (t['score']) for t in data]
    # y = np.array(y)
    y = np.reshape(y, (-1,1))

    return X, y

X, y = load_dbscore_data()

sc = StandardScaler()
X_transform = sc.fit_transform(X)

# y = mx + c

import statsmodels.api as sm
X_const = sm.add_constant(X_transform)

model = sm.OLS(y, X_const)
ls = model.fit()

print(ls.summary())

ls_c = ls.params[0]
ls_m1 = ls.params[1]
ls_m2 = ls.params[2]
ls_m3 = ls.params[3]
# print("[",ls_m1, ls_m2, ls_m3, "]", ls_c)

y_pred1 = ls_m1*X_transform[:,0] + ls_c
y_pred2 = ls_m2*X_transform[:,1] + ls_c
y_pred3 = ls_m3*X_transform[:,2] + ls_c

fig,axs = plt.subplots(2,2)
axs[0,0].plot(X_transform[:,0],y,'o')
axs[0,0].plot([min(X_transform[:,0]), max(X_transform[:,0])], [min(y_pred1), max(y_pred1)], color='red')
axs[0,1].plot(X_transform[:,1],y,'o')
axs[0,1].plot([min(X_transform[:,1]), max(X_transform[:,1])], [min(y_pred2), max(y_pred2)], color='red')
axs[1,0].plot(X_transform[:,2],y,'o')
axs[1,0].plot([min(X_transform[:,2]), max(X_transform[:,2])], [min(y_pred3), max(y_pred3)], color='red')
# plt.show()

def gradient_descent_naive(X, y):

    epochs = 100000
    min_grad = 0.0001
    learning_rate = 0.001
    
    m = [0.0, 0.0, 0.0]
    c = 0.0
    
    n = len(y)
    
    c_grad = 0.0
    m_grad = 0.0
    
    for epoch in range(epochs):
        
        for i in range(n):
            y_pred = m @ X[i] + c
            m_grad += 2*(y_pred-y[i]) * X[i]
            c_grad += 2*(y_pred - y[i])

        c_grad /= n
        m_grad /= n
        
        m = m - learning_rate * m_grad
        c = c - learning_rate * c_grad
        
        if ( epoch % 1000 == 0):
            # print("epoch %d: m_grad=%f, c_grad=%f, m=%f, c=%f" %(epoch, m_grad, c_grad, m, c) )   
            print("epoch", epoch, ": m_grad=", m_grad, "m=", m, "c=", c)
        
        if ( all(abs(m_grad) < min_grad) and all(abs(c_grad) < min_grad) ):
            break
        
    return m, c

start_time = time.time()
m, c = gradient_descent_naive(X_transform, y)
end_time = time.time()

print("%f seconds" %(end_time - start_time))

print("\n\nFinal:")
print("gdn_m=[%f %f %f], gdn_c=%f" %(m[0], m[1], m[2], c) )
print("ls_m=[%f %f %f], ls_c=%f" %(ls_m1, ls_m2, ls_m3, ls_c) )


def gradient_descent_vectorized(X, y):
    epochs = 100000
    min_grad = 0.0001
    learning_rate = 0.001
    
    m = 0.0
    c = 0.0
    
    n = len(y)
    
    c_grad = 0.0
    m_grad = 0.0

    for epoch in range(epochs):    
    
        y_pred = m * X + c
        m_grad = (2*(y_pred - y)*X).sum(axis=0)/n
        

        c_grad = (2 * (y_pred - y)).sum()/n
        
        m = m - learning_rate * m_grad
        c = c - learning_rate * c_grad        

        if ( epoch % 1000 == 0):
            # print("epoch %d: m_grad=%f, c_grad=%f, m=%f, c=%f" %(epoch, m_grad, c_grad, m, c) )
            print("epoch", epoch, ": m_grad=", m_grad, "m=", m, "c=", c)
    
        # if ( abs(m_grad) < min_grad and abs(c_grad) < min_grad ):
        if ( all(abs(m_grad) < min_grad) and abs(c_grad) < min_grad ):
            break

    return m, c

start_time = time.time()
m, c = gradient_descent_vectorized(X_transform, y)
end_time = time.time()

print("%f seconds" %(end_time - start_time))

print("\n\nFinal:")
# print("gdv_m=%f, gdv_c=%f" %(m, c) )
print("gdv_m=[%f %f %f], gdv_c=%f" %(m[0], m[1], m[2], c) )
print("ls_m=[%f %f %f], ls_c=%f" %(ls_m1, ls_m2, ls_m3, ls_c) )

plt.show()