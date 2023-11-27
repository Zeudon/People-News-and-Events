import pandas as pd
import numpy as np

df = pd.read_csv(r'..\Data\Facebook\TimesNowFBRheaChak.csv',header=0)
df=df[['comment_count','like_count','level']]

df=df[ df['level'] == 1]
df.reset_index(inplace=True,drop=True)
count=0

count= len(df[ df['like_count'] < df['comment_count']])

print(count)
