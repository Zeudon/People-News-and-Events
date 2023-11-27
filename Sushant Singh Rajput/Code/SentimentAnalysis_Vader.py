from datetime import date
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd

# YouTube
# message_col_name='snippet.topLevelComment.snippet.textDisplay'
# date_col_name='snippet.topLevelComment.snippet.publishedAt'

# Facebook
message_col_name='message'
date_col_name='created_time'

data=pd.read_csv(r'..\Data\Facebook\TimesNowFBRheaChak.csv',header=0)

# Only for Facebook to separate Top Level Comments (level=1) from Replies (level=2)
data=data[data['level']==1]
data=data.reset_index(drop=True)

data=data[[message_col_name,date_col_name]]
data[message_col_name]=[str(i) for i in data[message_col_name]]

def sentiment_scores(message):
    obj=SentimentIntensityAnalyzer()

    sentiment_dict=obj.polarity_scores(message)
    return sentiment_dict

for i in range(len(data)):
    if(data[message_col_name][i]=='nan' or data[message_col_name][i]==""):
        data=data.drop(i)
data=data.reset_index(drop=True)

neg_scores=[]
pos_scores=[]
neu_scores=[]
comp_scores=[]
for i in range(len(data)):
    dict=sentiment_scores(data.at[i,message_col_name])
    pos_scores.append(dict['pos'])
    neg_scores.append(dict['neg'])
    neu_scores.append(dict['neu'])
    comp_scores.append(dict['compound'])

data['positive']=pos_scores
data['negative']=neg_scores
data['neutral']=neu_scores
data['compound']=comp_scores

data=data.sort_values(by=[date_col_name], ascending=True)

data.to_csv(r'..\Sentiment Vader\Facebook\Top Level Comments\TimesNowFBRheaChak_Sentiment.csv',header=True,index=False)

