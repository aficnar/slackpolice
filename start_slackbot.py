"""
This module starts a Slack bot that monitors the discussions and warns users
if they go off-topic.
"""

import nltk
import numpy as np
import os
import pandas as pd
import re
import spacy
import sys
import time
 
from slackclient import SlackClient
from __future__ import division

from pyemd import emd
from sklearn.metrics import euclidean_distances
from sklearn.metrics import accuracy_score

from nltk.corpus import stopwords
from nltk.tokenize import WordPunctTokenizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix

import db_connect # Connect to SQL



# ------------------------------------------------------------------------------
# Get the data
# ------------------------------------------------------------------------------

#Connect to SQL and pull the comments from all the subreddits we have:
all_subreddits = pd.read_sql("SELECT id FROM main_subreddits", 
							 con)['id'].tolist()
all_subreddit_names = pd.read_sql("SELECT name FROM main_subreddits", 
								  con)['name'].tolist()
data = []
for subreddit_id in all_subreddits:
    sql_query = ("""SELECT content FROM main_comments 
    				WHERE subreddit_id = '%s'""" % subreddit_id)
    data_local = pd.read_sql(sql_query, con)['content'].tolist()
    data.append(data_local)

# Take only the first 5000 from each channel:
total_size = 5000
data = [datum[:total_size] for datum in data]
lengths = [len(datum) for datum in data]



# ------------------------------------------------------------------------------
# Preprocess the text
# ------------------------------------------------------------------------------

# Get rid of the Emojis:
emoji_pattern = re.compile(
    u"(\ud83d[\ude00-\ude4f])|"  # emoticons
    u"(\ud83c[\udf00-\uffff])|"  # symbols & pictographs (1 of 2)
    u"(\ud83d[\u0000-\uddff])|"  # symbols & pictographs (2 of 2)
    u"(\ud83d[\ude80-\udeff])|"  # transport & map symbols
    u"(\ud83c[\udde0-\uddff])"   # flags (iOS)
    "+", flags = re.UNICODE)

# Load stopping words:
stopset = set(stopwords.words('english'))
my_stopset = set(['would', 'http', 'also', 'com', 'https']).union(stopset)

def clean_post(post):
    """
    Takes a post, cleans it and returns list of tokens.
    """
    if type(post) == str: conv_post = unicode(post, "utf-8")
    else: conv_post = post
    u_post = emoji_pattern.sub('', conv_post)
    u_post_let = re.sub("[^a-zA-Z]", " ", u_post) # Only words
    tokens = WordPunctTokenizer().tokenize(u_post_let)
    clean = [token.lower() for token in tokens if token.lower() 
    		 not in my_stopset and len(token) > 2]
    return clean

# Tokenize our datasets:
data_tokenized = [[clean_post(comment) for comment in datum] for datum in data]

# Initialize Spacy's language pipeline
vectorizer = CountVectorizer(stop_words = None)
nlp = spacy.load('en')

pos_set = set([u'NOUN', u'ADJ', u'VERB'])
def oov_checker_plus(s):
    """
    Check if a word is in Spacy's vocabulary AND is either a noun, a verb 
    or an adjective 
    """
    if type(s) == str: s = unicode(s, "utf-8")
    is_oov = nlp(s)[0].is_oov
    is_info_word = nlp(s)[0].pos_ in pos_set
    return (not is_oov) and is_info_word

# Only include the tokens that satisfy this, and rejoin them into messages:
data_tokenized = [[[t for t in tokens if oov_checker_plus(t) == True] 
                    for tokens in datum] for datum in data_tokenized]
data_clean = [[" ".join(tokens) for tokens in datum] 
			  for datum in data_tokenized]

# Let's divide our corpus in a training set (70%), 
# validation set (20%) and a test set (10%):
train_size = int(0.7 * total_size)
valid_size = int(0.2 * total_size)
test_size = int(0.1 * total_size)

data_train = [datum[:train_size] for datum in data_clean]
data_valid = [datum[train_size:train_size + valid_size] for datum in data_clean]
data_test = [datum[train_size + valid_size:train_size + valid_size + test_size] 
			 for datum in data_clean]

# Let's get rid of the empty messages in the training and the validation sets:
data_train = [filter(None, datum) for datum in data_train]
data_valid = [filter(None, datum) for datum in data_valid]



# ------------------------------------------------------------------------------
# Get word distributions
# ------------------------------------------------------------------------------

# In this section we'll use the bag of words to get the word distributions 
# for each channel and make the representative posts from the training data:
freqs_list = []
for datum in data_train:
    vectorizer = CountVectorizer(analyzer = "word", tokenizer = None,    
                                 preprocessor = None, stop_words = None, 
                                 max_features = 5000)
    this_bow = vectorizer.fit_transform(datum)
    this_bow_array = this_bow.toarray()
    vocabulary = vectorizer.get_feature_names()
    counts = np.sum(this_bow_array, axis = 0)
    top_words = pd.DataFrame({'word': vocabulary, 'count': counts})
    top_words = top_words.sort_values(by = 'count', ascending = False)
    top_series = top_words['count']
    top_series.index = top_words['word']
    freqs_list.append(top_series)
rel_freq_list = [f / f.sum() for f in freqs_list]



# ------------------------------------------------------------------------------
# Modified Word Mover's Distance
# ------------------------------------------------------------------------------

def wmd_mod(s1, i_cat, no_top):
    """
    Gives the modified WMD distance (see my blog for more info) between the 
    input string s1 and a representative message of no_top words of channel 
    with index i_cat
    """
    # Representative message made up of no_top top words:
    s2 = " ".join(rel_freq_list[i_cat].index.tolist()[:no_top])
    vect_fit = vectorizer.fit([s1, s2])
    spacy_words = nlp(" ".join(vect_fit.get_feature_names())) 
    # Make bag-of-words vectors 
    v_1, v_2 = vect_fit.transform([s1, s2])
    v_1 = v_1.toarray().ravel().astype(np.double)
    v_2 = v_2.toarray().ravel().astype(np.double)
    # For the representative message, get weights from word distributions:
    for i_s in range(len(v_2)):
        if str(spacy_words[i_s]) in rel_freq_list[i_cat][:no_top]:
            v_2[i_s] = rel_freq_list[i_cat][str(spacy_words[i_s])]
        else: 
            v_2[i_s] = 0
    v_1 /= v_1.sum()
    v_2 /= v_2.sum()    
    w2v_vectors = [w.vector for w in spacy_words]
    dist_matrix = euclidean_distances(w2v_vectors).astype(np.double)
    return emd(v_1, v_2, dist_matrix)

# Optimal number of top words (from the main_analysis.ipynb)
no_top_optimal = 180

# Optimal threshold
thresh_opt = 0.0535



# ------------------------------------------------------------------------------
# Slackbot code
# ------------------------------------------------------------------------------

# Our Slack officer is called Newman and here are his credentials:
bot_token = os.environ["SLACKBOT_TOKEN_NEWMAN"]
bot_name = 'officer_newman'

# Initialize the Slack client and find out its ID 
# (so we can filter its messages):
slack_client = SlackClient(bot_token)
users = slack_client.api_call("users.list").get('members')
for user in users:
    if 'name' in user and user.get('name') == bot_name:
        bot_id = user.get('id')

# Let's find the channel ID's and select the relevant ones:
channel_list = slack_client.api_call("channels.list")['channels']
all_channel_ids = [c['id'] for c in channel_list if 'ex' in c['name']]
all_channel_names = [c['name'] for c in channel_list if 'ex' in c['name']]

def parse_slack_output(slack_rtm_output):
    """
    Check if the output from Slack came from a user as a text message
    """
    output_list = slack_rtm_output
    if output_list and len(output_list) > 0:
        for output in output_list:
            if output and 'text' in output and 'user' in output:
                return (output['text'].strip().lower(), 
                		output['channel'], output['user'])
    return None, None, None

def handle_input(input_string, channel, user):
    """
    Handle the input and decide whether the bot reacts (and how) or not
    """
    # Clean the input string
    input_tokenized = clean_post(input_string)
    input_tokenized = [t for t in input_tokenized 
    				   if oov_checker_plus(t) == True]
    input_clean = " ".join(input_tokenized)
    generic = False
    # If it's empty after cleaning, it's generic
    if len(input_clean) == 0: 
        generic = True
    else:
        # Calculate WMD between the message and all the channels and find the
        # shortest one:
        wmd_avgs = [wmd_mod(input_clean, i, no_top_optimal) 
        			for i in range(len(data_train))]
        index_min = wmd_avgs.index(min(wmd_avgs))
        predicted_channel = all_channel_ids[index_min]
        top_indices = np.argsort(wmd_avgs)
        # If the relative difference between the top score and the next one is 
        # less than the threshold, flag as generic:
        top_score = wmd_avgs[top_indices[0]]
        next_score = wmd_avgs[top_indices[1]]
        rel_diff = np.abs(top_score - next_score) / top_score
        if rel_diff < thresh_opt: generic = True
        # React if the message is non-generic and the user is in the wrong
        # channel:
        if predicted_channel != channel and user != bot_id and generic == False:
            response = ("Hey <@" + user + ">, consider posting this in the <#" + 
                        predicted_channel + "> channel.")
            slack_client.api_call("chat.postMessage", channel = channel,
                                  text = response, as_user = True)

# Open the Slack RTM firehose:
if slack_client.rtm_connect():
    print("officer_newman connected and monitoring...")
    while True:
        command, channel, user = parse_slack_output(slack_client.rtm_read())
        if command and channel:
            handle_input(command, channel, user)
        time.sleep(1)
else:
    print("Connection failed.")