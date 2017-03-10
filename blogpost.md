
We love [Slack](https://slack.com/), a popular messaging app especially amongst developers. Slack brings communication together in one place. It's real-time messaging, archiving and search for teams. Conversations are organized in channels, often by topic. *Add examples* More often than not, channel names are insufficient to understand a given channel's topic. Veterans just *know* where to post content, new members of the team may struggle and post messages in the wrong channels. Nobody wants to be that nagging veteran team member trying to direct the rookie to a more appropriate place—let a bot do the work! I build a bot that learns the topics of different Slack channels, monitors conversations, and warns users when they go off topic. Give it a [try](https://slack-police.slack.com)!

At its heart, the bot needs to be able to compare two messages or documents; the user's input and messages already present in a channel (concatenated into one long document). A standard way to compare two documents is to use [bag-of-words](https://en.wikipedia.org/wiki/Bag-of-words_model) (BoW), which includes approaches such as [tf-idf](https://en.wikipedia.org/wiki/Tf%E2%80%93idf), and cosine similarity. Alas, BoW does not capture semantic properties of words, problems arise when documents share related but not identical words (e.g., press and media).

##Word Mover's Distance for Document Similarity
I used the [Word Mover's Distance](http://jmlr.org/proceedings/papers/v37/kusnerb15.pdf), a novel similarity metric built on top of and leveraging [word embeddings](https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf). Word embeddings are high dimensional representations of words. Words of similar meaning live close to one another in this higher dimensional space; word embeddings capture semantric properties of words (i.e., distributional semantics). I use pre-trained word embeddings from [Spacy](https://spacy.io/) trained on the Common Crawl corpus.

A natural way to estimate how dissimilar, or distant, two documents are is to look at the distance between the corresponding word vectors and, roughly speaking, add up those distances. That is the idea behind the Word Mover's Distance, and neatly, it is an instance of the well-known [Earth Mover's Distance](https://en.wikipedia.org/wiki/Earth_mover's_distance) (EMD) optimization problem, only formulated in the word embedding space. 

###Earth Mover's Distance for Vector Similarity
The EMD assumes that one has two collections of vectors, let's call them the <i>senders</i> and the <i>receivers</i>, and a matrix of their pair-wise distances. In addition to this, each of the vectors has a weight, which is a real number smaller than 1, indicating how much "goods" each of the sender vectors has to send and how much of the goods each of the receiver vectors needs to receive. The sum of all the weights for the receiver vectors is normalized to 1, as it is for the sender vectors. The problem now is, given the distances (costs) between the sender-receiver pairs, to determine the most efficient way to <i>move</i> the goods from the senders to the receivers, allowing for partial sending and receiving (i.e. so that a sender can send a portion of its goods to one receiver and another portion to another receiver). This is obviously a non-trivial constrained optimization problem, and it has a known solution, which can be easily implemented in Python with the <a href="https://pypi.python.org/pypi/pyemd"><code class="highlighter-rouge">pyemd</code></a> package. <br><br>

WMD is the application of the EMD problem to the context of word embeddings, where the senders and receivers are word embeddings of words from the first and second document we're comparing, respectively. The weights of the vectors are chosen to be proportional to the number of times the corresponding word appears in the document. The distances between the vectors are then calculated using standard Euclidean distances in the word embedding space. In this way we can easily calculate the WMD distance between two documents using the <code class="highlighter-rouge">pyemd</code> package.

###<i>O(p^3 log(p))</i>, Terrible Time Complexity
A practical obstacle in applying this method to our case is the fact that the EMD algorithm has a horrible time complexity: <i>O(p^3 log(p))</i>, where <i>p</i> is the number of unique words in the two documents. We would need to compare the user's input to all of the previous messages in all the channels, calculate the average distance for each of the channels, and the one with the smallest average distance would be our prediction for the channel to which the user's message should go. If the user posted the message in the channel we predicted, the bot doesn't do anything, otherwise the bot will advise the user to consider posting it to the predicted channel. For Slack teams that contain a lot of messages spread out over a lot of channels, this will not be a feasible approach for a real time response of the bot. 

So, let's modity the WMD. Comparing the input message to *all* the messages in a given channel seems excessive: surely there are messages that are more "representative" of the channel content than the others, and it's likely enough to compare the user input to those messages only. However, this would require expensive preprocessing, in which we essentially have to sort the channel messages using WMD as a key. But can we somehow *construct* a single message representative of an entire channel? 

###Slack Channel "Fingerprints"
Intuitively, this could be achieved by looking at word distributions in a given channel, as shown on the right. Obviously, to a person, looking at the first, say, 10 or so words that occur most often in a channel would give a pretty good idea of what that channel is about. A single message representative of that channel should therefore contain only those 10 words. To use this message in EMD / WMD, we need to choose the weights (see previous subsection) of the vectors representing the words in it. Since the weights in a standard WMD are directly proportional to how many times a given word appears in a message, we can make the weights in our representative message proportional to the number of times a given word appears in the entire channel (and then normalize it). 

In this way we've constructed a single representative message for each channel, and we only need to calculate the WMD distances between the input message and each of the representative messages, find the shortest one, and predict the corresponding channel as the one the input message is supposed to go to. 

Is 10 words enough to form a representative message? How about 30? 100? We can find the optimal number of words,  <i>n_words</i>, by treating it as a hyperparameter and tuning it on a validation set? Yes. The answer is 180.

##Model Performance
Finally, finally we need to compute the accuracy of our model. On the right we see the confusion matrix for the test set, showing the distributions of messages from their true categories over the predicted ones. The accuracy of this model is about 74%, which is pretty good, and a noticeable improvement from 68% that one gets from the tf-idf approach and using the cosine similarity as the metric. 

###Turns Out "Thank You's" Can Be Annoying 
In the confusion matrix we see some expected confusion with the closely related topics: for example, 24% of messages from the machine learning channel got misclasified as data science. If we look under the hood, we can see that a lot of these messages are in fact pretty generic (e.g. "thank you") and could belong to any channel. In fact, our model picked up on that: the distances to all the channels for these messages are pretty similar, and it just happens that the distance to the data science channel was the shortest one. 

Let's elimiate some of these "generic" messages, let's introduce a threshold: when the distance between the channel the message was posted in and the channel that the message was classified to belong to is smaller than some value epsilon, we'll ignore the prediction of the model and the bot won't advise the user. To keep things simple, we will use a fixed, relative threshold for the entire corpus. We'll treat the threshold as a hyperparameter, and tune it on the validation set. <a onclick="showhide('expl2')">Click here for more details.</a>

However, the messages are not labeled as generic or non-generic; we cannot code up some automatic verification process that can tell us how accurately the model is performing in flagging messages as generic (for a given value of the threshold). We would need an actual human being to look at the example the model flagged as generic and decide if it is indeed generic. That seems cumbersome! Let's take the following practical approach of maximizing the accuracy likelihood.

In order not to decrease the accuracy of our model too much, we would like to minimize the number of correctly classified messages that are flagged as generic, as flagging a message as generic introduces a possibility that we mis-flagged it, which would decrease the accuracy of the model. On the other hand, in order to try to increase the accuracy of our model, we would like to maximize the number of incorrectly classified messages flagged as generic, as flagging a message as generic also introduces a possibility that we correctly flagged it, which would increase the accuracy of the model. As we increase the threshold, the amount of correctly classified messages predicted to be generic will increase, while the amount of the incorrectly classified messages predicted to be non-generic will decrease, as shown in the plot above. A natural choice for the optimal threshold is the place where the two curves intersect, which is about 0.05 in our case.

Now that we have chosen an optimal threshold, we can apply our model to the test set, and then manually check whether all the messages our model flagged as generic are indeed generic, and use that to update the effective accuracy of the model. <b>This results in the final accuracy of about 84% (which is better than the initial *insert*).

@andrej add brief summary of project's technical solution

###Data
Slack data is hard to come by, since it's private. The next best thing is [Reddit](https://www.reddit.com/), since its data is easily available and has a similar structure to Slack, where instead of channels, different topics are grouped into subreddits. 

For the purposes of demonstrating the model, I chose the following five topics (subreddits): *Diving*, *Handball*, *Corgi*, *Data Science*, and *Machine Learning*. These have been chosen intentionally so that some of them are more similar to each other and others are less (plus, they also tell you something about the things I like!). 

The relevant data (submissions and comments) can then be downloaded using Reddit's excellent API through an easy-to-use [`PRAW`](https://praw.readthedocs.io/en/latest/#) package for Python, and stored in a SQL database. See the code on my Github for details.

---

# <a name="demo">3. Demo: Officer Slackbot in action</a>

To showcase my bot's might, I made a [demo Slack team](https://slack-police.slack.com) -- go ahead and try it out! I created a generic user with a **username `slack.police.demo@gmail.com`** and the password I have either shared with you when I presented the project or you can [email me](mailto:aficnar@gmail.com) to ask for it.

In the demo Slack team I created 5 channels, corresponding to the 5 subreddits above, and populated those channels with the comments obtained from the corresponding subreddits. For simplicity, I focused only on comments, rather than the submissions, since they tend to be shorter, perhaps more faithfully mimicking the form of Slack messages. 

To upload the Reddit data to my Slack team, I first registered 4 [bot users](https://api.slack.com/bot-users) on Slack (posing as famous characters on Seinfeld!), and used the excellent package [`slackclient`](https://github.com/slackapi/python-slackclient) that allows one to communicate with Slack's API from Python. For more details on how to build simple bots in Python, check out my code here on Github and / or have a look at a great tutorial from the [Full Stack Python](https://www.fullstackpython.com/blog/build-first-slack-bot-python.html) blog. The bot itself is hosted on [AWS](https://aws.amazon.com/), constantly monitoring the discussions in the demo Slack team.

Below is a little illustration of bot's basic functionality, showing me entering a couple of messages in the *Diving* channel. As you can see, as long as the messages are vaguely related to diving, or are of generic content that could belong to any of the channels (e.g. 'thank you', etc.), the bot doesn't bother me. But if I mention something more closely related to one of the other existing channels, the bot will let me know where those messages might be more appropriate. 
<p><img src="images/usage_animation.gif" width="700px" hspace="20" vspace="20" align="center"></p>

---



# <a name="summary">6. Summary & what more can be done</a>

In summary, I've built a user-friendly and a reasonably smart bot for Slack that helps users stay on topic. 

This bot prototype can be obviously applied to platforms other than Slack, including Reddit and Stack Overflow. It can be also potentially developed into more advanced applications, including automatic email classification and perhaps even filtering out hate speech on Twitter. 

This is a project I built in three weeks at Insight. That's not a lot of time, but it's been a lot of fun, and I've got a bunch of ideas I would love to implement some time in the near future:



### Better estimate of bot's performance in the real world <a onclick="showhide('extra1')"><font size="2">[show / hide]</font></a>
<div style="display:none" id="extra1"><br>To better estimate the performance of the bot in the real world, we need more data. One idea would be to download something like a 100 random subreddits and then:
<ul>
<li>Choose a sub-corpus of, say 5, random subreddits.</li>
<li>Calculate the accuracy of the model with the threshold (upper bound on the actual accuracy) and without it (lower bound). </li>
<li>Do this for several more random choices of 5 subreddits, and average.</li>
<li>Repeat the whole procedure for different sub-corpus sizes, and report the lower and upper band on the accuracy as a function of the number of channels considered. </li>
</ul>
</div>



### Starting the bot on an empty channel <a onclick="showhide('extra2')"><font size="2">[show / hide]</font></a><br>
<div style="display:none" id="extra2"><br>The bot obviously works well when there's a lot of messages already present in Slack channels. But what if the Slack team is just starting and there's not that much data? Cold start is a known problem in machine learning, and our modification of the WMD approach can be useful in this context. For example, we could simply ask the users to list the keywords in the description of a channel in their order of importance, and then use these keywords to form representative messages for each channels, inferring the weights based on the order of the words. Then, after enough messages have been posted (“enough” under a certain criterion), the bot can start using them for prediction.</div>



### More features <a onclick="showhide('extra3')"><font size="2">[show / hide]</font></a><br>
<div style="display:none" id="extra3"><br>In the current model, I've been simply looking at the textual content of the messages, but in the real world there are other features we could use that could help us in correctly classifying the messages. 
<ul>
<li>One possible feature is the identity of the user posting the message. Intuitively, if this is a veteran user, we don't expect them to go off-topic on a channel, but a new user might and we should pay more attention. We could implement that by e.g. giving a different weight to the messages from different users, based on the amount of text they entered before. Or, we could simply introduce a higher threshold for the more "trusted" users.</li>
<li>Another potentially relevant feature is the time a user posts a message. More recent messages should be more important, and this could be taken into account by e.g. weighing them with time-dependent exponentials. This would also allow for the channel topics to slowly change over time.</li>
</ul></div>



### Augment a pre-trained word2vec <a onclick="showhide('extra4')"><font size="2">[show / hide]</font></a>
<div style="display:none" id="extra4"><br>Word2vec representations of words used in the current version of the bot come from a pre-trained model, i.e. the linguistic context for those words is inferred from the generic Common crawl texts. But the relations between the words may change for specific channels. For example, the words <i>random</i> and <i>forest</i> may seem pretty dissimilar in a general context, but if we're in a machine learning channel, their vectors should be pretty close. It would be therefore great to try to augment the pre-trained model with the examples from the current channels, so that we retain the power of the pre-trained model and its wealth of word vectors, but also teach it to recognize relations between words specific to our channels.</div>



### Density functions in w2v space <a onclick="showhide('extra5')"><font size="2">[show / hide]</font></a>
<div style="display:none" id="extra5"><br>Grouping messages into different classes using their representation in a higher dimensional space sounds a lot like clustering. What if we define a simple density function in the word embedding space for each channel, made up of Gaussians centered around each word's vector, with a tune-able variance. Then, when a user enters a message, we could simply calculate its value in each of the density functions, and predict based on which of the density functions has the highest value (and tune the variance on the validation set). 
<ul>
<li>This would also perhaps allow for better flagging of the generic messages, as their clusters should have a similar support in all the density functions.</li>
<li>Finally, this approach could allow for detecting when a similar parallel topic arises in a channel. In this case, we would see the original cluster slowly splitting into two clusters close to each other, and when the bot notices this, it could advise the Slack team owner to consider splitting the channel into two.</li>
</ul></div>



### Making the bot more user friendly <a onclick="showhide('extra6')"><font size="2">[show / hide]</font></a>
<div style="display:none" id="extra6">
<br>It would be great to improve the overall user experience of the bot, for example:
<ul>
<li>Maybe sometimes you don't want the bot to bother you, so it'd nice to be able to shut it down. This can be done by e.g. implementing a <a href="https://api.slack.com/slash-commands">slash command</a> on Slack. Those essentially send an <code class="highlighter-rouge">HTTP POST</code> request to a URL of choice, which can be used to start and pause the bot.</li>
<li>With the same slash command we could also modify bot's parameters. For example, if the bot is too sensitive, we could increase its threshold.</li>
<li>It would be perhaps better that, when the bot advises users, it does so privately, i.e. so that only the affected user can see it, so it doesn't create additional noise on the channel. These kinds of hidden messages are possible, but for that purposes the bot would need to be registered as a Slack App.</li>
<li>Having the bot as a Slack app would also allow it to display buttons. When the bot suggests that the message is more appropriate for a different channel, the user could click on 'okay' and the message would get copied over to that channel. If the user wants to keep the message in the channel where it was posted, he or she could click 'no', which the bot could use for online learning (for example, for updating its thresholds).</li>
</ul></div>



### Other approaches <a onclick="showhide('extra7')"><font size="2">[show / hide]</font></a><br>
<div style="display:none" id="extra7">
<br>In this project I started with a simple bag-of-words / tf-idf approach, after which I focused on applying (and modifying) Word Mover's Distance to quantify how similar (or distant) two documents are. There are other models on the market, and it would be interesting to see the performance of some of those in this context:
<ul>
<li><a href="https://github.com/facebookresearch/fastText">FastText</a> is a recent method from Facebook AI, essentially an extension of the standard word2vec that represents words as bags of characters n-grams. It can be trained really quickly and allows for online learning.</li>
<li>The current version of the bot constructs representative messages using a simple bag-of-words approach, but it would be interesting to try and see if topic extraction via <a href="https://en.wikipedia.org/wiki/Latent_Dirichlet_allocation">LDA</a> would improve accuracy. </li>
<li>Other potentially interesting models to consider would be <a href="https://github.com/ryankiros/skip-thoughts">skip-thoughts</a>, <a href="https://radimrehurek.com/gensim/models/doc2vec.html">doc2vec</a> and perhaps even training an <a href="https://en.wikipedia.org/wiki/Long_short-term_memory">LSTM</a> neural network.</li>
</ul></div>

---

## <a name="about_me">7. About me</a>
My name is Andrej Ficnar, I'm a theoretical physicist and an aspiring data scientist. I got my PhD from Columbia University in 2014, where I studied applications of string theory to real-world physical systems in order to better understand their properties. After that, I moved to the University of Oxford for a postdoc, where I got more and more interested in applying my coding and analytical skills to data-rich environments. This eventually led me back to New York in 2016, and then into Insight. 

Check out my profile on [Linkedin](https://www.linkedin.com/in/aficnar) and my codes on [Github](https://github.com/aficnar/) for more info about me.

<script>
function showhide(elem) {
    var x = document.getElementById(elem);
    if (x.style.display === 'none') {
        x.style.display = 'block';
    } else {
        x.style.display = 'none';
    }
}
</script>
