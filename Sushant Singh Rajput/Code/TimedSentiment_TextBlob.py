import pandas as pd
import re, string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer 
from textblob import TextBlob
import matplotlib.pyplot as plt

sw = stopwords.words('english')
lemmatizer = WordNetLemmatizer()

# YouTube Columns
message_col_name='snippet.topLevelComment.snippet.textDisplay'
time_col_name='snippet.topLevelComment.snippet.publishedAt'

#Facebook Columns
# message_col_name='message'
# time_col_name='created_time'

data=pd.read_csv(r'..\Sentiment\Facebook\Timed Sentiment\IndiaTodayYT_TimedSentiment.csv',header=0)

# Only for Facebook to separate Top Level Comments (level=1) from Replies (level=2)
# data=data[data['level']==1]
# data=data.reset_index(drop=True)

# Extract messages and created time, convert to string
data=data[[message_col_name,time_col_name]]
data[message_col_name]=[str(i) for i in data[message_col_name]]

def lem(text):
    text = [lemmatizer.lemmatize(t) for t in text]
    text = [lemmatizer.lemmatize(t, 'v') for t in text]
    return text

def clean_text(text):
    text = text.lower()
    text = re.sub('@', '', text)
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = re.sub(r"[^a-zA-Z ]+", "", text) 
    #Tokenize the data
    text = nltk.word_tokenize(text)
    #Remove stopwords
    text = [w for w in text if w not in sw]
    return text

# Clean and lemmatize the text, remove empty comments
data[message_col_name] = data[message_col_name].apply(lambda x: clean_text(x))
data[message_col_name] = data[message_col_name].apply(lambda x: lem(x))

# Remove empty messages
for i in range(len(data)):
    if(data[message_col_name][i]==['nan'] or data[message_col_name][i]==[]):
        data=data.drop(i)
data=data.reset_index(drop=True)

#Get sentiment from comments
data[message_col_name] = [str(thing) for thing in data[message_col_name]]
sentiment = []
for i in range(len(data)):
    blob = TextBlob(data[message_col_name][i])
    for sentence in blob.sentences:
        sentiment.append(sentence.sentiment.polarity)
data['sentiment']=sentiment

data=data.sort_values(by=[time_col_name], ascending=True)

data.to_csv(r'.\IndiaTodayFB_TimedSentiment.csv',header=True)

