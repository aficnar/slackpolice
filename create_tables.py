"""
This module creates the tables in our main SQL database that will store the
data from Reddit (subreddits, submissions and comments).
"""

import pandas as pd
import db_connect # Connect to SQL

cur = con.cursor()

cur.execute("""
            CREATE TABLE main_subreddits
            (
            id VARCHAR(255) NOT NULL,
            name TEXT NOT NULL,
            title TEXT NOT NULL, 
            created INT NOT NULL,
            total_submissions INT NOT NULL
            )
            """)

cur.execute("""
            CREATE TABLE main_submissions
            (
            id VARCHAR(255) NOT NULL,
            subreddit_id VARCHAR(255) NOT NULL,
            content TEXT
            )
            """)

cur.execute("""
            CREATE TABLE main_comments
            (
            id VARCHAR(255) NOT NULL,
            subreddit_id VARCHAR(255) NOT NULL,
            submission_id VARCHAR(255) NOT NULL,
            content TEXT
            )
            """)

con.commit()
cur.close()
con.close()