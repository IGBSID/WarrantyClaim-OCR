from sqlmodel import create_engine
import os

server = os.environ['SQL_DB_SERVER']
database = os.environ['SQL_DB_DATABASE']
username = os.environ['SQL_DB_USERNAME']
password = os.environ['SQL_DB_PASSWORD']
driver = os.environ['SQL_DB_DRIVER']

connection_string = 'Driver=' + driver + ';Server=tcp:' + server + ',1433;Database=' + database + ';Uid=' + username + ';Pwd=' + password + ';Encrypt=yes;TrustServerCertificate=no;Connection Timeout=30;'
from sqlalchemy import create_engine
import urllib

params = urllib.parse.quote_plus(connection_string)
conn_str = 'mssql+pyodbc:///?odbc_connect={}'.format(params)
engine = create_engine(conn_str,echo=True)

print('connection is ok')