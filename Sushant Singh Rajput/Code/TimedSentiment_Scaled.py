import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# YouTube Columns
# message_col_name='snippet.topLevelComment.snippet.textDisplay'
# time_col_name='snippet.topLevelComment.snippet.publishedAt'

# Facebook Columns
message_col_name='message'
time_col_name='created_time'

data=pd.read_csv(r'..\Sentiment\Sentiment Vader\Facebook\Top Level Comments\TimesNowFBRheaChak_Sentiment.csv',header=0)

# Convert time to date format, calculate minutes since first comment
# Facebook Date Format
data[time_col_name]=pd.to_datetime(data[time_col_name],format='%Y-%m-%dT%H:%M:%S+0000')
# YouTube Date Format
# data[time_col_name]=pd.to_datetime(data[time_col_name],format='%Y-%m-%dT%H:%M:%SZ')

start_date=data[time_col_name][0]
time_since_col=[((i-start_date).total_seconds())/60 for i in data[time_col_name]]
data['time_since_col']=time_since_col

# Finding mean every delta minutes
time=0
delta=5
pos_sum=0
neg_sum=0
neu_sum=0
comp_sum=0
num=0
mean_pos=[]
mean_neg=[]
mean_neu=[]
mean_comp=[]
for i in range(len(data)):
    if(data.at[i,'time_since_col'] <= float(time+delta)):
        pos_sum+=data.at[i,'positive']
        neg_sum+=data.at[i,'negative']
        neu_sum+=data.at[i,'neutral']
        comp_sum+=data.at[i,'compound']
        num+=1
    else:
        mean_pos.append(pos_sum/num)
        mean_neg.append(neg_sum/num)
        mean_neu.append(neu_sum/num)
        mean_comp.append(comp_sum/num)
        pos_sum=data.at[i,'positive']
        neg_sum=data.at[i,'negative']
        neu_sum=data.at[i,'neutral']
        comp_sum=data.at[i,'compound']
        num=1
        time+=delta
mean_pos.append(pos_sum/num)
mean_neg.append(neg_sum/num)
mean_neu.append(neu_sum/num)
mean_comp.append(comp_sum/num)

x = range(len(mean_pos))
y1 = mean_pos
y2 = mean_neg
y4 = mean_comp

# Positive and negative proportion in each comment
plt.plot(x,y1,label="Positive Proportion")
plt.plot(x,y2,label="Negative Proportion")
plt.title('Comment Sentiment Composition')
plt.xlabel(f'Time (mean sentiment of comments taken every {delta} minutes)')
plt.ylabel('Proportion')
plt.legend()
plt.show()

# Scatter plot and trend line
plt.scatter(x, y4)
plt.title('Comments Polarity Trend')
plt.xlabel(f'Time (mean sentiment of comments taken every {delta} minutes)')
plt.ylabel('Compound Score')
z = np.polyfit(x, y4, 1)
p = np.poly1d(z)
plt.plot(x,p(x),"r--",label="Trend")
plt.legend()
plt.show()