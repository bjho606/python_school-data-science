import pymysql
import numpy as np


def classification_performance_eval(y, y_predict):
    tp, tn, fp, fn = 0,0,0,0
    
    for y, yp in zip(y,y_predict):
        if y == 1 and yp == 1:
            tp += 1
        elif y == 1 and yp == -1:
            fn += 1
        elif y == -1 and yp == 1:
            fp += 1
        else:
            tn += 1
            
    accuracy = (tp+tn)/(tp+tn+fp+fn)
    precision = (tp)/(tp+fp)
    recall = (tp)/(tp+fn)
    f1_score = 2*precision*recall / (precision+recall)
    
    return accuracy, precision, recall, f1_score



conn = pymysql.connect(host='localhost', user='root', password='chunjay606', db='data_science')
curs = conn.cursor(pymysql.cursors.DictCursor)

sql = "select * from iris"
curs.execute(sql)

data  = curs.fetchall()
# print(data)

curs.close()
conn.close()

X = [ (t['SepalLengthCm'], t['SepalWidthCm'], t['PetalLengthCm'], t['PetalWidthCm'] ) for t in data ]
X = np.array(X)
# print(X.shape)

y = [ 1 if (t['Species'] == 'Iris-versicolor') else -1 for t in data]
y = np.array(y)
# print(y)
# print(y.shape)


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.33, random_state=42)

from sklearn import tree

dtree = tree.DecisionTreeClassifier()

dtree_model = dtree.fit(X_train, y_train)

y_predict = dtree_model.predict(X_test)

acc, prec, rec, f1 = classification_performance_eval(y_test, y_predict)

'''
print("accuracy=%f" %acc)
print("precision=%f" %prec)
print("recall=%f" %rec)
print("f1_score=%f" %f1)
'''

from sklearn.model_selection import KFold

accuracy = []
precision = []
recall = []
f1_score = []

kf = KFold(n_splits=5, random_state=42, shuffle=True)

for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    dtree = tree.DecisionTreeClassifier()
    dtree = dtree_model.fit(X_train, y_train)
    y_predict = dtree_model.predict(X_test)
    acc, prec, rec, f1 = classification_performance_eval(y_test, y_predict)
    accuracy.append(acc)
    precision.append(prec)
    recall.append(rec)
    f1_score.append(f1)

# print(accuracy)

import statistics

'''
print("average_accuracy =", statistics.mean(accuracy))
print("average_precision =", statistics.mean(precision))
print("average_recall =", statistics.mean(recall))
print("average_f1_score =", statistics.mean(f1_score))
'''
