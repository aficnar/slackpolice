"""
This module displays all the relevant tables and columns in our SQL database.
"""

import pandas as pd
import db_connect # Connect to SQL

# Get the list of tables
sql_query = """
            SELECT table_name 
            FROM information_schema.tables
            WHERE table_name LIKE 'main_%'
            """
table_list = pd.read_sql(sql_query, con)

# Get the list of all the columns
sql_query = """
            SELECT column_name, table_name 
            FROM information_schema.columns
            WHERE table_name LIKE 'main_%'
            """
col_list = pd.read_sql(sql_query, con)

# Generate a dictionary containing all column names
DF_dict = {}
for name in list(table_list['table_name']):
    DF_dict[name] = list(col_list[col_list['table_name'] 
                         == name]['column_name'])

# Fill it up so all lists are equal length
max_length = max([len(f) for f in DF_dict.values()])
for key, value in DF_dict.items():
    DF_dict[key] = DF_dict[key] + ['-----'] * (max_length - len(DF_dict[key]))

# Create a DataFrame and print it
all_columns = pd.DataFrame(DF_dict)
print all_columns

con.commit()
con.close()