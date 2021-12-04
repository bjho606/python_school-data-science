import pandas as pd

xl_file = '/Users/JaehoByun/JB/_School/2021_2 데이터사이언스/과제및시험/score.xlsx'
df = pd.read_excel(xl_file)
# print(df)

midterm20 = df['midterm'] >= 20
final20 = df['final'] >= 20

df2 = df[midterm20 & final20]
print(df2[['sno','midterm','final']])