import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# YouTube Columns
message_col_nameYT='snippet.topLevelComment.snippet.textDisplay'
time_col_nameYT='snippet.topLevelComment.snippet.publishedAt'
# message_col_nameYT='Message'
# time_col_nameYT='Time'

#Facebook Columns
message_col_nameFB='message'
time_col_nameFB='created_time'

data1=pd.read_csv(r"..\SSR\Sentiment\Sentiment Vader\YouTube\Top Level Comments\TimesNowYTRheaChak_Sentiment.csv",header=0)
data2=pd.read_csv(r"..\SSR\Sentiment\Sentiment Vader\Facebook\Top Level Comments\TimesNowFBRheaChak_Sentiment.csv",header=0)

# Convert time to date format, calculate minutes since first comment
# Facebook Date Format
# data1[time_col_nameFB]=pd.to_datetime(data1[time_col_nameFB],format='%Y-%m-%dT%H:%M:%S+0000')
data2[time_col_nameFB]=pd.to_datetime(data2[time_col_nameFB],format='%Y-%m-%dT%H:%M:%S+0000')
# YouTube Date Format
data1[time_col_nameYT]=pd.to_datetime(data1[time_col_nameYT],format='%Y-%m-%dT%H:%M:%SZ')
# data2[time_col_nameFB]=pd.to_datetime(data2[time_col_nameFB],format='%Y-%m-%dT%H:%M:%SZ')

start_date=[data1[time_col_nameYT][0],data2[time_col_nameFB][0]]
if (start_date[0]<start_date[1]):
    time_since_col1=[((i-start_date[0]).total_seconds())/60 for i in data1[time_col_nameYT]]
    time_since_col2=[((i-start_date[0]).total_seconds())/60 for i in data2[time_col_nameFB]]
else:
    time_since_col1=[((i-start_date[1]).total_seconds())/60 for i in data1[time_col_nameYT]]
    time_since_col2=[((i-start_date[1]).total_seconds())/60 for i in data2[time_col_nameFB]]

data1['time_since_col']=time_since_col1
data2['time_since_col']=time_since_col2
# sentiments1=[data1['positive'],data1['negative'],data1['neutral'],data1['compound'],data1[time_col_nameYT]]
# sentiments2=[data2['positive'],data2['negative'],data2['neutral'],data2['compound'],data2[time_col_nameYT]]
sentiments1=[data1['positive'],data1['negative'],data1['neutral'],data1['compound'],data1["time_since_col"]]
sentiments2=[data2['positive'],data2['negative'],data2['neutral'],data2['compound'],data2["time_since_col"]]

x1=sentiments1[-1]
x2=sentiments2[-1]
x=[x1,x2]
y1 = [sentiments1[0],sentiments2[0]]
y2 = [sentiments1[1],sentiments2[1]]
y4 = [sentiments1[3],sentiments2[3]]

plt.plot(x[0],y1[0],label="Times Now Rhea Chakraborty YouTube")
plt.plot(x[1],y1[1],label="Times Now Rhea Chakraborty Facebook")
plt.title('Proportion of Positive Sentiment in Comments')
plt.xlabel(f'Minutes since first comment')
plt.ylabel('Proportion')
plt.xlim(0,4000)
plt.legend()
plt.savefig(r"..\SSR\Plots\Superimposed\Individual\Without Outliers\TimesNowRheaChakYT_TimesNowRheaChakFB_Positive.pdf")
plt.show()

plt.plot(x[0],y2[0],label="Times Now Rhea Chakraborty YouTube")
plt.plot(x[1],y2[1],label="Times Now Rhea Chakraborty Facebook")
plt.title('Proportion of Negative Sentiment in Comments')
plt.xlabel(f'Minutes since first comment')
plt.ylabel('Proportion')
plt.xlim(0,4000)
plt.legend()
plt.savefig(r"..\SSR\Plots\Superimposed\Individual\Without Outliers\TimesNowRheaChakYT_TimesNowRheaChakFB_Negative.pdf")
plt.show()

# Scatter plot and trend line
plt.scatter(x[0], y4[0],color='red',label="Times Now Rhea Chakraborty YouTube")
plt.scatter(x[1], y4[1],color="blue",label="Times Now Rhea Chakraborty Facebook")
plt.title('Comments Polarity Trend')
plt.xlabel(f'Minutes since first comment')
plt.ylabel('Compound Score')
plt.xlim(0,4000)
z = [np.polyfit(x[0], y4[0], 1),np.polyfit(x[1], y4[1], 1)]
p1 = np.poly1d(z[0])
p2 = np.poly1d(z[1])
plt.plot(x[0],p1(x[0]),"r--")
plt.plot(x[1],p2(x[1]),'b--')
plt.legend()
plt.savefig(r"..\SSR\Plots\Superimposed\Individual\Without Outliers\TimesNowRheaChakYT_TimesNowRheaChakFB_Trend.pdf")
plt.show()