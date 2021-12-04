import pymysql
import numpy as np
# from sklearn.metrics import classification_report, accuracy_score, precision_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn import tree, svm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import statistics

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

    # print(tp, tn, fp, fn)

    accuracy = (tp+tn)/(tp+tn+fp+fn)
    precision = (tp)/(tp+fp) if (tp+fp) != 0 else 0
    recall = (tp)/(tp+fn) if (tp+fn) != 0 else 0
    f1_score = 2*precision*recall / (precision+recall) if precision != 0 else 0
    
    return accuracy, precision, recall, f1_score

# [Algorithm 1] Decision Tree model classification
def decision_tree_model(X_train, X_test, y_train, y_test):
    classifier = tree.DecisionTreeClassifier()
    dtree_model = classifier.fit(X_train, y_train)

    y_predict = dtree_model.predict(X_test)
    # print(y_test)
    # print(y_predict)

    acc, prec, rec, f1 = classification_performance_eval(y_test, y_predict)

    return acc, prec, rec, f1

# [Algorithm 2] SVM(Support Vector Machine) model classification
def svm_model(X_train, X_test, y_train, y_test):
    # classifier = svm.SVC()
    classifier = svm.SVC(kernel='rbf', C=8, gamma=0.1)  # 여기를 조절해야댐
    svm_model = classifier.fit(X_train, y_train)

    y_predict = svm_model.predict(X_test)
    # print(y_test)
    # print(y_predict)
    
    acc, prec, rec, f1 = classification_performance_eval(y_test, y_predict)

    return acc, prec, rec, f1

# [Algorithm 3] Logistic Regression model classification
def logistic_regression_model(X_train, X_test, y_train, y_test):
    classifier = LogisticRegression()
    lg_model = classifier.fit(X_train, y_train)

    y_predict = lg_model.predict(X_test)
    # print(y_test)
    # print(y_predict)

    acc, prec, rec, f1 = classification_performance_eval(y_test, y_predict)

    return acc, prec, rec, f1

conn = pymysql.connect(host='localhost', user='root', password='chunjay606', db='data_science')
curs = conn.cursor(pymysql.cursors.DictCursor)

sql = "select * from db_score"
curs.execute(sql)

data  = curs.fetchall()
# print(data)

curs.close()
conn.close()

X = [ (t['homework'], t['discussion'], t['final'] ) for t in data ]
X = np.array(X)
# print(X.shape)

# [Classification Type 1] Binary
y = [ 1 if (t['grade'] == 'B') else -1 for t in data]
y = np.array(y)
# print(y)
# print(y.shape)

# [Train & Test Dataset 1] Random Split
print()
print("------------------Train_Test_Split----------------------")
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.33, random_state=42)
# print(y_test)

# [Algorithm 1] Decision Tree model classification
acc, prec, rec, f1 = decision_tree_model(X_train, X_test, y_train, y_test)

print("< Decision Tree >")
print("accuracy=%f" %acc)
print("precision=%f" %prec)
print("recall=%f" %rec)
print("f1_score=%f" %f1)
print("")


# [Algorithm 2] SVM(Support Vector Machine) model classification
acc, prec, rec, f1 = svm_model(X_train, X_test, y_train, y_test)

print("< SVM (Support Vector Machine) >")
print("accuracy=%f" %acc)
print("precision=%f" %prec)
print("recall=%f" %rec)
print("f1_score=%f" %f1)
print("")


# [Algorithm 3] Logistic Regression model classification
acc, prec, rec, f1 = logistic_regression_model(X_train, X_test, y_train, y_test)

print("< Logistic Regression >")
print("accuracy=%f" %acc)
print("precision=%f" %prec)
print("recall=%f" %rec)
print("f1_score=%f" %f1)
print("")

print()
print("-------------------K-fold cross validation-------------------")
# [Train & Test Dataset 2] K-fold
dt_accuracy = []
dt_precision = []
dt_recall = []
dt_f1_score = []

s_accuracy = []
s_precision = []
s_recall = []
s_f1_score = []

lr_accuracy = []
lr_precision = []
lr_recall = []
lr_f1_score = []

kf = KFold(n_splits=5, random_state=42, shuffle=True)

for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    # 1. Decision Tree
    acc, prec, rec, f1 = decision_tree_model(X_train, X_test, y_train, y_test)

    dt_accuracy.append(acc)
    dt_precision.append(prec)
    dt_recall.append(rec)
    dt_f1_score.append(f1)

    # 2. SVM
    acc, prec, rec, f1 = svm_model(X_train, X_test, y_train, y_test)

    s_accuracy.append(acc)
    s_precision.append(prec)
    s_recall.append(rec)
    s_f1_score.append(f1)

    # 3. Logistic Regression
    acc, prec, rec, f1 = logistic_regression_model(X_train, X_test, y_train, y_test)

    lr_accuracy.append(acc)
    lr_precision.append(prec)
    lr_recall.append(rec)
    lr_f1_score.append(f1)

# print(accuracy)

print("< Decision Tree >")
print("average_accuracy =", statistics.mean(dt_accuracy))
print("average_precision =", statistics.mean(dt_precision))
print("average_recall =", statistics.mean(dt_recall))
print("average_f1_score =", statistics.mean(dt_f1_score))
print()

print("< SVM >")
print("average_accuracy =", statistics.mean(s_accuracy))
print("average_precision =", statistics.mean(s_precision))
print("average_recall =", statistics.mean(s_recall))
print("average_f1_score =", statistics.mean(s_f1_score))
print()

print("< Logistic Regression >")
print("average_accuracy =", statistics.mean(lr_accuracy))
print("average_precision =", statistics.mean(lr_precision))
print("average_recall =", statistics.mean(lr_recall))
print("average_f1_score =", statistics.mean(lr_f1_score))
print()