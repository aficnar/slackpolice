"""
This module downloads comments from Reddit and saves them to our SQL database. 
We choose the subreddits we're interested in, and set the maximal number of 
comments to download. We then connect via PRAW package to Reddit's API,
start downloading the comments, and report progress along the way.
"""

import os
import pandas as pd
import praw
import sys
import time

import db_connect # Connect to SQL

# Choose the subreddits to download and max comments to download
chosen_subreddits = ['diving', 'Handball', 'corgi', 'datascience', 
					 'MachineLearning']
comments_threshold = 5000

#Connect to Reddit's API
reddit = praw.Reddit(client_id = os.environ["PRAW_CLIENT_ID"],
                     client_secret = os.environ["PRAW_CLIENT_SECRET"],
                     password = os.environ["PRAW_PASSWORD"],
                     user_agent = 'slackbot by /u/aficnar',
                     username = 'aficnar')

# Get the relevant data for each of the subreddits and store them in SQL
for i in range(len(chosen_subreddits)):
    subreddit = reddit.subreddit(chosen_subreddits[i])
    db_tuple = (subreddit.id, chosen_subreddits[i], subreddit.title, 
                int(subreddit.created), 0)
    sql_query = """
    			INSERT INTO main_subreddits
    			VALUES (%s, %s, %s, %s, %s)
    			"""
    cur.execute(sql_query, db_tuple)

#Couple of functions that we'll use to display the remaining time

def stringify(num):
	"""
	Takes a number and returns a string putting a zero in front if it's 
	single digit.
	"""
    num_string = str(num)
    if len(num_string) == 1: num_string = '0' + num_string
    return num_string

def convert_secs(secs_float):
	"""
	Takes a number of seconds and returns a string that looks like a clock.
	"""
    secs = int(round(secs_float))
    sec_display = secs % 60
    mins = secs // 60
    mins_display = mins % 60
    hrs_display = mins // 60
    export_string = (stringify(hrs_display) + ':' + 
    				 stringify(mins_display) + ':' + 
    				 stringify(sec_display))
    return export_string

# Put the subreddits to be downloaded in pull_df DataFrame 
# (useful for adding other subreddits later on)
all_df = pd.read_sql("SELECT * FROM main_subreddits", con)
pull_df = all_df
pull_df.index = range(pull_df.shape[0])

#Loop over the above subreddits, submissions and comments and save them to SQL
for i in range(pull_df.shape[0]):
    t_start = time.time()
    print ("Subreddit " + str(i + 1) + " / " + 
    	   str(pull_df.shape[0]) + ": " + pull_df['title'][i])
    # Get relevant subreddit info
    subreddit = reddit.subreddit(pull_df['name'][i])
    subreddit_id = pull_df['id'][i]
    comment_count = 0
    # Loop over all the submissions
    for submission in subreddit.submissions():
        submission.comments.replace_more(limit = 0); # For deep comments
        submission_tuple = (submission.id, subreddit_id, submission.selftext)
        comment_tuples = []
        # Loop over all the comments in this submission
        comment_list = submission.comments.list()
        for comment in comment_list:
            comment_tuples.append((comment.id , subreddit_id, 
            					   submission.id, comment.body))
        if len(comment_tuples) > 0:
            comment_str = ','.join(cur.mogrify("(%s,%s,%s,%s)", x) 
            					   for x in comment_tuples)
            cur.execute("INSERT INTO main_comments VALUES " + comment_str)
        submission_query =  "INSERT INTO main_submissions VALUES (%s, %s, %s)"
        cur.execute(submission_query, submission_tuple)
        comment_count = comment_count + len(comment_list)
        # The rest is for pretty tracking of progress
        sys.stdout.write("\rComments processed: %d" % comment_count)
        sys.stdout.flush()
        # Exit if you've downloaded more than comments_threshold comments
        if comment_count > comments_threshold: break
    t_finish = time.time()
    print
    print 'Runtime: ' + convert_secs(t_finish - t_start)
    print("-" * 50)

#Check the total amount of submissions and comments downloaded:
all_comments = pd.read_sql("SELECT * FROM main_comments", con)
all_submissions = pd.read_sql("SELECT * FROM main_submissions", con)
print "Total number of comments downloaded:", all_comments.shape[0]
print "Total number of submissions downloaded:", all_submissions.shape[0]

con.commit()
cur.close()
con.close()