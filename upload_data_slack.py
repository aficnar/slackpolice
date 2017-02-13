"""
This module uploads the previously downloaded Reddit data to Slack.
There's a Slack team called `Slack Police`, with 5 bots there. 
The idea is to pick one of the first four bots here randomly as we upload the 
messages to Slack, so it looks like a real conversation. 
The last bot, officer_newman will be the one monitoring discussions and 
warning about users going off-topic.
"""

import os
import pandas as pd
import praw
import SlackClient
import time

import db_connect # Connect to SQL

# Initialize the Slack bot clients
bot_tokens = [os.environ["SLACKBOT_TOKEN_JERRY"],
              os.environ["SLACKBOT_TOKEN_ELAINE"],
              os.environ["SLACKBOT_TOKEN_COSMO"],
              os.environ["SLACKBOT_TOKEN_GEORGE"],
              os.environ["SLACKBOT_TOKEN_NEWMAN"]]
bot_names = ['jerry', 'elaine', 'cosmo', 'george', 'officer_newman']
slack_uploaders = [SlackClient(bot_tokens[i]) for i in range(4)]

# List of all the channels on our Slack team
channel_list = slack_uploaders[0].api_call("channels.list")['channels']

# Select the subreddits to be uploaded and the corresponding channels:
sql_query = "SELECT name FROM main_subreddits"
all_subreddit_upload_names = pd.read_sql(sql_query, con)['name'].tolist()
all_channel_upload_list = [channel_list[i]['id'] for i in 
						   range(len(all_subreddit_upload_names))]
subreddit_upload_names = all_subreddit_upload_names
channel_upload_list = all_channel_upload_list

# Loop over them and, for now, just upload the first 1000 comments, 
# Also also pause for 1s between each ping, due to Slack's API rate limit.
comment_limit = 1000
cnt = 0
for i_sub in range(len(subreddit_upload_names)):
    # Get the comments from a given subreddit
    sql_query = """
                SELECT content 
                FROM main_comments 
                WHERE subreddit_id = (
                    SELECT id 
                    FROM main_subreddits 
                    WHERE name = '%s'
                    )
                """ % subreddit_upload_names[i_sub]
    comments = pd.read_sql(sql_query, con)['content'].tolist()[:comment_limit]
    # Upload each comment to Slack as a random bot
    for c in comments:
        bot_index = random.randint(0,3)
        slack_uploaders[bot_index].api_call("chat.postMessage", 
        									channel=channel_upload_list[i_sub],
                                            text = c, as_user = True);
        time.sleep(1)
        cnt = cnt + 1
        output_string = ("\rTime remaining: " + convert_secs(comment_limit 
        				 * len(subreddit_upload_names) - cnt))
        sys.stdout.write(output_string)
        sys.stdout.flush()

con.commit()
con.close()