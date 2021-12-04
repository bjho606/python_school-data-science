import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

xl_file = '/Users/JaehoByun/JB/_School/2021_2 데이터사이언스/과제및시험/db_score.xlsx'
df = pd.read_excel(xl_file)
# print(df)

df_midterm = df['midterm']
df_final = df['final']
df_score = df['score']

# 1. Mean, Median
midterm_mean = np.mean(df_midterm)
midterm_median = np.median(df_midterm)
final_mean = np.mean(df_final)
final_median = np.median(df_final)
score_mean = np.mean(df_score)
score_median = np.median(df_score)
print("< Mean >")
print("midterm mean : " , midterm_mean)
print("final mean : " , final_mean)
print("score mean : " , score_mean)
print("-----------------------")
print("< Median >")
print("midterm median : " , midterm_median)
print("final median : " , final_median)
print("score median : " , score_median)
print("-----------------------")

# 2. Mode
mode_grade = stats.mode(df['grade'])
print("< Mode >")
print("grade mode : ", mode_grade[0])
print("-----------------------")

# 3. Variance, Standart Deviation
midterm_var = np.var(df_midterm)
midterm_stdev = np.std(df_midterm)
final_var = np.var(df_final)
final_stdev = np.std(df_final)
score_var = np.var(df_score)
score_stdev = np.std(df_score)
print("< Variance >")
print("midterm variance : " , midterm_var)
print("final variance : " , final_var)
print("score variance : " , score_var)
print("-----------------------")
print("< Standard Deviation")
print("midterm variance : ", midterm_stdev)
print("final variance : ", final_stdev)
print("score variance : ", score_stdev)
print("-----------------------")

# 4. Percentile
p = np.linspace(0, 100, 20)
y1 = np.percentile(df_midterm, p, interpolation='linear')
y2 = np.percentile(df_final, p, interpolation='linear')
y3 = np.percentile(df_score, p, interpolation='linear')

fig, ax = plt.subplots(1, 3)
fig.suptitle('< Percentile >')
fig.set_size_inches(10, 3)
ax[0].plot(p, y1)
ax[0].set_title("Midterm")

ax[1].plot(p, y2)
ax[1].set_title("Final")

ax[2].plot(p, y3)
ax[2].set_title("Score")

plt.subplots_adjust(wspace=0.5)
fig.tight_layout()
plt.show()

# 5. Boxplot
plt.title('< Boxplot >')
plt.ylim(-5, 100)
plt.boxplot([df_midterm, df_final, df_score], labels=['midterm', 'final', 'score'])
plt.show()

# 6. Histogram
fig, ax = plt.subplots(1, 3)
fig.suptitle('< Histogram >')
fig.set_size_inches(10, 3)
ax[0].hist(df_midterm)
ax[0].set_title("Midterm Histogram")

ax[1].hist(df_final)
ax[1].set_title("Final Histogram")

ax[2].hist(df_score)
ax[2].set_title("Score Histogram")

plt.subplots_adjust(wspace=0.5)
fig.tight_layout()
plt.show()

# 7. Scatter Plot
fig, ax = plt.subplots(1, 3, constrained_layout=True)
fig.suptitle('< Scatter Plot >')
fig.set_size_inches(10, 3)
ax[0].scatter(df_midterm, df_final)
ax[0].set_xlabel('Midterm')
ax[0].set_ylabel("Final")

ax[1].scatter(df_midterm, df_score)
ax[1].set_xlabel('Midterm')
ax[1].set_ylabel("Score")

ax[2].scatter(df_final, df_score)
ax[2].set_xlabel('Final')
ax[2].set_ylabel("Score")

plt.subplots_adjust(wspace=1)
fig.tight_layout()
plt.show()