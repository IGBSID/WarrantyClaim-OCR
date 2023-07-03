from sqlmodel import create_engine
import os
# from dotenv import load_dotenv
# from pathlib import Path

# print("os.environ.get('ENV_FILE_LOCATION')=" +os.environ['ENV_FILE_LOCATION'])
# dotenv_path = Path(os.environ.get('ENV_FILE_LOCATION'))
# # print("dotenv_path=", dotenv_path)
# load_dotenv(dotenv_path=dotenv_path)

# server = 'sql-in-cappa-d-domain-001.database.windows.net'
# database = 'automatedwarrantyclaim'
# username = 'automatedwarrantyclaim'
# password = 'd6a356bd-7d5b-479e-9c8c-1efec6752b6d'
# driver= '{ODBC Driver 18 for SQL Server}'
# connection_string = 'Driver={ODBC Driver 18 for SQL Server};Server=tcp:sql-in-cappa-d-domain-001.database.windows.net,1433;Database=automatedwarrantyclaim;Uid=automatedwarrantyclaim;Pwd=d6a356bd-7d5b-479e-9c8c-1efec6752b6d;Encrypt=yes;TrustServerCertificate=no;Connection Timeout=30;'
server = os.environ['SQL_DB_SERVER']
database = os.environ['SQL_DB_DATABASE']
username = os.environ['SQL_DB_USERNAME']
password = os.environ['SQL_DB_PASSWORD']
driver = os.environ['SQL_DB_DRIVER']

connection_string = 'Driver=' + driver + ';Server=tcp:' + server + ',1433;Database=' + database + ';Uid=' + username + ';Pwd=' + password + ';Encrypt=yes;TrustServerCertificate=no;Connection Timeout=30;'
# print("connection_string= " + connection_string)
# connection_string = 'Driver={ODBC Driver 18 for SQL Server};Server=tcp:sql-in-cappa-d-domain-001.database.windows.net,1433;Database=automatedwarrantyclaim;Uid=automatedwarrantyclaim;Pwd=d6a356bd-7d5b-479e-9c8c-1efec6752b6d;Encrypt=yes;TrustServerCertificate=no;Connection Timeout=30;'
# print("connection_string= " + connection_string)
from sqlalchemy import create_engine
import urllib

params = urllib.parse.quote_plus(connection_string)
conn_str = 'mssql+pyodbc:///?odbc_connect={}'.format(params)
engine = create_engine(conn_str,echo=True)

print('connection is ok')