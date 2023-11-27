from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import pandas as pd
import wordcloud

## Make sure words are present
def is_char(word):
    if('π' in word):
        return False
    for c in word:
        if(c>'z' or c<'A'):
            return False
    return True

def get_FB_comments(df,level):
    df=df[df['level']==level]
    return df


## File name
df = pd.read_csv(r'..\CSV\NDTVYTEnglish_Replies.csv',header=0)

words=""
stopwords=set(STOPWORDS)

## Column Name the text is in

## YouTube
col_name='snippet.textOriginal'

## Facebook
# col_name='message'
# df=get_FB_comments(df,2)

for comment in df[col_name]:
    comment=str(comment)
    tokens=comment.split()
    for i in range(len(tokens)):
        ## Remove Replies
        if(tokens[i][0]=='@'):  
            tokens[i]=''
            continue
        ## Lower case
        tokens[i]=tokens[i].lower()
        ## Remove non characters, nan
        if(tokens[i]=='nan' or (not(is_char(tokens[i])))):
            tokens[i]=''
    words+=' '.join(tokens)+' '

wordcloud=WordCloud(width=800,height=600,background_color='white',stopwords=stopwords,min_font_size=10).generate(words)
plt.figure(figsize=(8,8), facecolor=None)
plt.axis("off")
plt.tight_layout(pad=0)
plt.imshow(wordcloud)
plt.show()
