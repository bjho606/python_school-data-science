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
    tpA, fpA, fnA = 0,0,0
    tpB, fpB, fnB = 0,0,0
    tpC, fpC, fnC = 0,0,0

    for y, yp in zip(y, y_predict):
        if y == 0 and yp == 0:
            tpA += 1
        elif y == 1 and yp == 1:
            tpB += 1
        elif y == 2 and yp == 2:
            tpC += 1
        elif y == 0 and yp == 1:
            fnA += 1
            fpB += 1
        elif y == 1 and yp == 2:
            fnB += 1
            fpC += 1
        elif y == 2 and yp == 0:
            fnC += 1
            fpA += 1
        elif y == 0 and yp == 2:
            fnA += 1
            fpC += 1
        elif y == 1 and yp == 0:
            fnB += 1
            fpA += 1
        elif y == 2 and yp == 1:
            fnC += 1
            fpB += 1

    accuracy = (tpA+tpB+tpC)/(tpA+tpB+tpC+fpA+fpC+fnA+fnC+fpB+fpC)
    precision_A = (tpA)/(tpA+fpA) if (tpA+fpA) != 0 else 0
    recall_A = (tpA)/(tpA+fnA) if (tpA+fnA) != 0 else 0
    f1_score_A = 2*precision_A*recall_A / (precision_A+recall_A) if precision_A != 0 else 0
    precision_B = (tpB)/(tpB+fpB) if (tpB+fpB) != 0 else 0
    recall_B = (tpB)/(tpB+fnB) if (tpB+fnB) != 0 else 0
    f1_score_B = 2*precision_B*recall_B / (precision_B+recall_B) if precision_B != 0 else 0
    precision_C = (tpC)/(tpC+fpC) if (tpC+fpC) != 0 else 0
    recall_C = (tpC)/(tpC+fnC) if (tpC+fnC) != 0 else 0
    f1_score_C = 2*precision_C*recall_C / (precision_C+recall_C) if precision_C != 0 else 0
    
    return accuracy, precision_A, recall_A, f1_score_A, precision_B, recall_B, f1_score_B, precision_C, recall_C, f1_score_C

# [Algorithm 1] Decision Tree model classification
def decision_tree_model(X_train, X_test, y_train, y_test):
    classifier = tree.DecisionTreeClassifier()
    dtree_model = classifier.fit(X_train, y_train)

    y_predict = dtree_model.predict(X_test)
    # print(y_test)
    # print(y_predict)

    acc, precA, recA, f1A, precB, recB, f1B, precC, recC, f1C = classification_performance_eval(y_test, y_predict)

    return acc, precA, recA, f1A, precB, recB, f1B, precC, recC, f1C

# [Algorithm 2] SVM(Support Vector Machine) model classification
def svm_model(X_train, X_test, y_train, y_test):
    # classifier = svm.SVC()
    classifier = svm.SVC(kernel='rbf', C=8, gamma=0.1)  # 여기를 조절해야댐
    svm_model = classifier.fit(X_train, y_train)

    y_predict = svm_model.predict(X_test)
    # print(y_test)
    # print(y_predict)
    
    acc, precA, recA, f1A, precB, recB, f1B, precC, recC, f1C = classification_performance_eval(y_test, y_predict)

    return acc, precA, recA, f1A, precB, recB, f1B, precC, recC, f1C

# [Algorithm 3] Logistic Regression model classification
def logistic_regression_model(X_train, X_test, y_train, y_test):
    classifier = LogisticRegression(max_iter=1000)
    lg_model = classifier.fit(X_train, y_train)

    y_predict = lg_model.predict(X_test)
    # print(y_test)
    # print(y_predict)

    acc, precA, recA, f1A, precB, recB, f1B, precC, recC, f1C = classification_performance_eval(y_test, y_predict)

    return acc, precA, recA, f1A, precB, recB, f1B, precC, recC, f1C

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

# [Classification Type 2] Multi-Class
y = []
for t in data:
    if t['grade'] == 'A':
        y.append(0)
    elif t['grade'] == 'B':
        y.append(1)
    else:
        y.append(2)
y = np.array(y)
# print(y)
# print(y.shape)

# [Train & Test Dataset 1] Random Split
print()
print("------------------Train_Test_Split----------------------")
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.33, random_state=42)
# print(y_test)

# [Algorithm 1] Decision Tree model classification
acc, precA, recA, f1A, precB, recB, f1B, precC, recC, f1C = decision_tree_model(X_train, X_test, y_train, y_test)

print("< Decision Tree >")
print("accuracy=%f" %acc)
print()
print("A precision=%f" %precA)
print("A recall=%f" %recA)
print("A f1_score=%f" %f1A)
print("B precision=%f" %precB)
print("B recall=%f" %recB)
print("B f1_score=%f" %f1B)
print("C precision=%f" %precC)
print("C recall=%f" %recC)
print("C f1_score=%f" %f1C)
print("")


# [Algorithm 2] SVM(Support Vector Machine) model classification
acc, precA, recA, f1A, precB, recB, f1B, precC, recC, f1C = svm_model(X_train, X_test, y_train, y_test)

print("< SVM (Support Vector Machine) >")
print("accuracy=%f" %acc)
print()
print("A precision=%f" %precA)
print("A recall=%f" %recA)
print("A f1_score=%f" %f1A)
print("B precision=%f" %precB)
print("B recall=%f" %recB)
print("B f1_score=%f" %f1B)
print("C precision=%f" %precC)
print("C recall=%f" %recC)
print("C f1_score=%f" %f1C)
print("")


# [Algorithm 3] Logistic Regression model classification
acc, precA, recA, f1A, precB, recB, f1B, precC, recC, f1C = logistic_regression_model(X_train, X_test, y_train, y_test)

print("< Logistic Regression >")
print("accuracy=%f" %acc)
print()
print("A precision=%f" %precA)
print("A recall=%f" %recA)
print("A f1_score=%f" %f1A)
print("B precision=%f" %precB)
print("B recall=%f" %recB)
print("B f1_score=%f" %f1B)
print("C precision=%f" %precC)
print("C recall=%f" %recC)
print("C f1_score=%f" %f1C)
print("")

print()
print("-------------------K-fold cross validation-------------------")
# [Train & Test Dataset 2] K-fold
dt_accuracy = []
dt_precisionA = []
dt_recallA = []
dt_f1_scoreA = []
dt_precisionB = []
dt_recallB = []
dt_f1_scoreB = []
dt_precisionC = []
dt_recallC = []
dt_f1_scoreC = []

s_accuracy = []
s_precisionA = []
s_recallA = []
s_f1_scoreA = []
s_precisionB = []
s_recallB = []
s_f1_scoreB = []
s_precisionC = []
s_recallC = []
s_f1_scoreC = []

lr_accuracy = []
lr_precisionA = []
lr_recallA = []
lr_f1_scoreA = []
lr_precisionB = []
lr_recallB = []
lr_f1_scoreB = []
lr_precisionC = []
lr_recallC = []
lr_f1_scoreC = []

kf = KFold(n_splits=5, random_state=42, shuffle=True)

for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    # 1. Decision Tree
    acc, precA, recA, f1A, precB, recB, f1B, precC, recC, f1C = decision_tree_model(X_train, X_test, y_train, y_test)

    dt_accuracy.append(acc)
    dt_precisionA.append(precA)
    dt_recallA.append(recA)
    dt_f1_scoreA.append(f1A)
    dt_precisionB.append(precB)
    dt_recallB.append(recB)
    dt_f1_scoreB.append(f1B)
    dt_precisionC.append(precC)
    dt_recallC.append(recC)
    dt_f1_scoreC.append(f1C)

    # 2. SVM
    acc, precA, recA, f1A, precB, recB, f1B, precC, recC, f1C = svm_model(X_train, X_test, y_train, y_test)

    s_accuracy.append(acc)
    s_precisionA.append(precA)
    s_recallA.append(recA)
    s_f1_scoreA.append(f1A)
    s_precisionB.append(precB)
    s_recallB.append(recB)
    s_f1_scoreB.append(f1B)
    s_precisionC.append(precC)
    s_recallC.append(recC)
    s_f1_scoreC.append(f1C)

    # 3. Logistic Regression
    acc, precA, recA, f1A, precB, recB, f1B, precC, recC, f1C = logistic_regression_model(X_train, X_test, y_train, y_test)

    lr_accuracy.append(acc)
    lr_precisionA.append(precA)
    lr_recallA.append(recA)
    lr_f1_scoreA.append(f1A)
    lr_precisionB.append(precB)
    lr_recallB.append(recB)
    lr_f1_scoreB.append(f1B)
    lr_precisionC.append(precC)
    lr_recallC.append(recC)
    lr_f1_scoreC.append(f1C)

# print(accuracy)

print("< Decision Tree >")
print("average_accuracy =", statistics.mean(dt_accuracy))
print()
print("A_average_precision =", statistics.mean(dt_precisionA))
print("A_average_recall =", statistics.mean(dt_recallA))
print("A_average_f1_score =", statistics.mean(dt_f1_scoreA))
print("B_average_precision =", statistics.mean(dt_precisionB))
print("B_average_recall =", statistics.mean(dt_recallB))
print("B_average_f1_score =", statistics.mean(dt_f1_scoreB))
print("C_average_precision =", statistics.mean(dt_precisionC))
print("C_average_recall =", statistics.mean(dt_recallC))
print("C_average_f1_score =", statistics.mean(dt_f1_scoreC))
print()

print("< SVM >")
print("average_accuracy =", statistics.mean(s_accuracy))
print()
print("A_average_precision =", statistics.mean(s_precisionA))
print("A_average_recall =", statistics.mean(s_recallA))
print("A_average_f1_score =", statistics.mean(s_f1_scoreA))
print("B_average_precision =", statistics.mean(s_precisionB))
print("B_average_recall =", statistics.mean(s_recallB))
print("B_average_f1_score =", statistics.mean(s_f1_scoreB))
print("C_average_precision =", statistics.mean(s_precisionC))
print("C_average_recall =", statistics.mean(s_recallC))
print("C_average_f1_score =", statistics.mean(s_f1_scoreC))
print()

print("< Logistic Regression >")
print("average_accuracy =", statistics.mean(lr_accuracy))
print()
print("A_average_precision =", statistics.mean(lr_precisionA))
print("A_average_recall =", statistics.mean(lr_recallA))
print("A_average_f1_score =", statistics.mean(lr_f1_scoreA))
print("B_average_precision =", statistics.mean(lr_precisionB))
print("B_average_recall =", statistics.mean(lr_recallB))
print("B_average_f1_score =", statistics.mean(lr_f1_scoreB))
print("C_average_precision =", statistics.mean(lr_precisionC))
print("C_average_recall =", statistics.mean(lr_recallC))
print("C_average_f1_score =", statistics.mean(lr_f1_scoreC))
print()