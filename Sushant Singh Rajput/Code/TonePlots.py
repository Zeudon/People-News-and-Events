import pandas as pd
import matplotlib.pyplot as plt
import collections
import numpy as np

def getTones(data):
    tones={'Analytical':0,'Anger':0,'Confident':0,'Fear':0,'Joy':0,'Sadness':0,'Tentative':0}
    total=0
    for i in range(len(data)):
        str = data.at[i,'tone']
        start=str.find("'tone_name'")
        limit=str.find("'sentences_tone'")
        if(limit==-1):
            limit=len(str)
        flag=False
        while(start!=-1 and start<limit):
            flag=True
            start+=len("'tone_name': '")
            end=str.find("'",start)
            tone=str[start:end]
            if(tone in tones.keys()):
                tones[tone]+=1
            else:
                tones[tone]=1
            start=str.find("'tone_name'",end)
        if(flag):
            total+=1
    tones={k: v/total for k,v in tones.items()}
    tones=collections.OrderedDict(sorted(tones.items()))
    return tones

data1=pd.read_csv(r"..\SSR\Tone\YouTube\Top Level Comments\TimesNowYTRheaChak_Tone.csv",header=0)
data2=pd.read_csv(r"..\SSR\Tone\Facebook\Top Level Comments\TimesNowFBRheaChak_Tone.csv",header=0)

tones1=getTones(data1)
tones2=getTones(data2)
# tones=getTones(data1)
br=np.arange(len(tones1.keys()))

plt.title('Tones of the comments')
# plt.bar(tones.keys(),tones.values())
plt.bar(br-0.2, tones1.values(), 0.4, label='YouTube',color='r')
plt.bar(br+0.2, tones2.values(), 0.4, label='Facebook',color='b')
plt.xticks(br,tones1.keys())
plt.xlabel('Tones')
plt.ylabel('Proportion of comments')
plt.legend()
plt.savefig(r"..\SSR\Plots\Superimposed\Tone\TimesNowRheaChak.pdf")
plt.show()
