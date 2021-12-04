import pymysql
import pandas as pd
 
def load_dbscore_data():
    xl_file = '/Users/JaehoByun/JB/_School/2021_2 데이터사이언스/과제및시험/db_score_3_labels.xlsx'
    db_score = pd.read_excel(xl_file)

    conn = pymysql.connect(host='localhost', user='root', password='chunjay606', db='data_science')
    curs = conn.cursor(pymysql.cursors.DictCursor)
        
    drop_sql = """drop table if exists db_score """
    curs.execute(drop_sql)
    conn.commit()
    
    import sqlalchemy
    database_username = 'root'
    database_password = 'chunjay606'
    database_ip       = 'localhost'
    database_name     = 'data_science'
    database_connection = sqlalchemy.create_engine('mysql+pymysql://{0}:{1}@{2}/{3}'.
                                                   format(database_username, database_password, 
                                                          database_ip, database_name))
    db_score.to_sql(con=database_connection, name='db_score', if_exists='replace')      

if __name__ == '__main__':    
    load_dbscore_data()  
