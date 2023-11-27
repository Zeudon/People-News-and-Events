import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

def getSentiment(data,delta):
    time=data.at[0,'time_since_col']
    pos_sum=0
    neg_sum=0
    neu_sum=0
    comp_sum=0
    time_sum=0
    num=0
    mean_pos=[]
    mean_neg=[]
    mean_neu=[]
    mean_comp=[]
    mean_time=[]
    for i in range(len(data)):
        if(data.at[i,'time_since_col'] <= float(time+delta)):
            pos_sum+=data.at[i,'positive']
            neg_sum+=data.at[i,'negative']
            neu_sum+=data.at[i,'neutral']
            comp_sum+=data.at[i,'compound']
            time_sum+=data.at[i,'time_since_col']
            num+=1
        else:
            if(num==0):
                mean_pos.append(0)
                mean_neg.append(0)
                mean_neu.append(0)
                mean_comp.append(0)
                mean_time.append(mean_time[-1]+delta/2)
            else:
                mean_pos.append(pos_sum/num)
                mean_neg.append(neg_sum/num)
                mean_neu.append(neu_sum/num)
                mean_comp.append(comp_sum/num)
                mean_time.append(time_sum/num)
            pos_sum=data.at[i,'positive']
            neg_sum=data.at[i,'negative']
            neu_sum=data.at[i,'neutral']
            comp_sum=data.at[i,'compound']
            time_sum=data.at[i,'time_since_col']
            num=1
            time+=delta
    mean_time.append(time_sum/num)
    mean_pos.append(pos_sum/num)
    mean_neg.append(neg_sum/num)
    mean_neu.append(neu_sum/num)
    mean_comp.append(comp_sum/num)
    return [mean_pos,mean_neg,mean_neu,mean_comp,mean_time]

# YouTube Columns
# message_col_nameYT='snippet.topLevelComment.snippet.textDisplay'
# time_col_nameYT'snippet.topLevelComment.snippet.publishedAt'

#Facebook Columns
message_col_nameFB='message'
time_col_nameFB='created_time'

data1=pd.read_csv(r'..\Sentiment\Sentiment Vader\Facebook\Top Level Comments\TimesNowFB_Sentiment.csv',header=0)
data2=pd.read_csv(r'..\Sentiment\Sentiment Vader\Facebook\Top Level Comments\TimesNowFBRheaChak_Sentiment.csv',header=0)

# Convert time to date format, calculate minutes since first comment
# Facebook Date Format
data1[time_col_nameFB]=pd.to_datetime(data1[time_col_nameFB],format='%Y-%m-%dT%H:%M:%S+0000')
data2[time_col_nameFB]=pd.to_datetime(data2[time_col_nameFB],format='%Y-%m-%dT%H:%M:%S+0000')
# YouTube Date Format
# data1[time_col_nameYT]=pd.to_datetime(data1[time_col_nameYT],format='%Y-%m-%dT%H:%M:%SZ')
# data2[time_col_nameYT]=pd.to_datetime(data2[time_col_nameYT],format='%Y-%m-%dT%H:%M:%SZ')

# Finding mean every delta minutes
delta=30

start_date=[data1[time_col_nameFB][0],data2[time_col_nameFB][0]]
if (start_date[0]<start_date[1]):
    time_since_col1=[((i-start_date[0]).total_seconds())/60 for i in data1[time_col_nameFB]]
    time_since_col2=[((i-start_date[0]).total_seconds())/60 for i in data2[time_col_nameFB]]
else:
    time_since_col1=[((i-start_date[1]).total_seconds())/60 for i in data1[time_col_nameFB]]
    time_since_col2=[((i-start_date[1]).total_seconds())/60 for i in data2[time_col_nameFB]]

data1['time_since_col']=time_since_col1
data2['time_since_col']=time_since_col2
sentiments1=getSentiment(data1,delta)
sentiments2=getSentiment(data2,delta)

x1=sentiments1[-1]
x2=sentiments2[-1]

x=[x1,x2]
y1 = [sentiments1[0],sentiments2[0]]
y2 = [sentiments1[1],sentiments2[1]]
y4 = [sentiments1[3],sentiments2[3]]

plt.plot(x[0],y1[0],label="Times Now")
plt.plot(x[1],y1[1],label="Times Now Rhea Chakraborty")
plt.title('Proportion of Positive Sentiment in Comments')
plt.xlabel(f'Minutes since first comment (mean sentiment taken every {delta} minutes)')
plt.ylabel('Proportion')
# plt.xlim(0,4000)
plt.legend()
plt.savefig(r'..\Plots\Facebook\Timed Sentiment\Vader\Superimposed\Synched\30 Mins\With Outliers\TimesNow_TimesNow_RheaChakPositive.pdf')
plt.show()

plt.plot(x[0],y2[0],label="Times Now")
plt.plot(x[1],y2[1],label="Times Now Rheak Charaborty")
plt.title('Proportion of Negative Sentiment in Comments')
plt.xlabel(f'Minutes since first comment (mean sentiment taken every {delta} minutes)')
plt.ylabel('Proportion')
# plt.xlim(0,4000)
plt.legend()
plt.savefig(r'..\Plots\Facebook\Timed Sentiment\Vader\Superimposed\Synched\30 Mins\With Outliers\TimesNow_TimesNow_RheaChakNegative.pdf')
plt.show()

# Scatter plot and trend line
plt.scatter(x[0], y4[0],color='red',label='Times Now')
plt.scatter(x[1], y4[1],color='blue',label='Times Now Rhea Chakraborty')
plt.title('Comments Polarity Trend')
plt.xlabel(f'Minutes since first comment (mean sentiment taken every {delta} minutes)')
plt.ylabel('Compound Score')
z = [np.polyfit(x[0], y4[0], 1),np.polyfit(x[1], y4[1], 1)]
p1 = np.poly1d(z[0])
p2 = np.poly1d(z[1])
plt.plot(x[0],p1(x[0]),"r--")
plt.plot(x[1],p2(x[1]),'b--')
# plt.xlim(0,4000)
plt.legend()
plt.savefig(r'..\Plots\Facebook\Timed Sentiment\Vader\Superimposed\Synched\30 Mins\With Outliers\TimesNow_TimesNow_RheaChakTrend.pdf')
plt.show()