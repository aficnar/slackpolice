
"""
This module connects to the SQL database on Amazon RDS. 
"""

import psycopg2
import os

dbname = 'slack_db'
host = 'slack-sql.cwwptmjl07ap.us-west-2.rds.amazonaws.com'
port = '5432'
user = os.environ["SQL_USERNAME"]
password = os.environ["SQL_PASSWORD"]

con = psycopg2.connect(
   database = dbname,
   user = user,
   password = password,
   host = host,
   port = port
)