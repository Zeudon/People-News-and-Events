from nltk.util import trigrams
import pandas as pd
import re, string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer 
from textblob import TextBlob
import matplotlib.pyplot as plt

sw = stopwords.words('english')
message_col_name='snippet.textOriginal'
lemmatizer = WordNetLemmatizer()
data=pd.read_csv(r'..\Data\YouTube\IndiaTodayYTRheaChak.csv',header=0)


# Only for Facebook to separate Top Level Comments (level=1) from Replies (level=2)
# data=data[data['level']==2]
# data=data.reset_index(drop=True)


data=data[[message_col_name]]
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

for i in range(len(data)):
    if(data[message_col_name][i]==['nan'] or data[message_col_name][i]==[]):
        data=data.drop(i)
data=data.reset_index(drop=True)

all_words=[]        
for i in range(len(data)):
    all_words = all_words + data[message_col_name][i]

# Frequency
nlp_words = nltk.FreqDist(all_words)
plot1 = nlp_words.plot(20, color='salmon', title='Word Frequency')

# Bigrams
bigram=list(nltk.bigrams(all_words))
words_2=nltk.FreqDist(bigram)
words_2.plot(20,color='salmon',title='Bigram Frequency')

# Trigrams
trigram=list(nltk.trigrams(all_words))
words_3=nltk.FreqDist(trigram)
words_3.plot(20,color='salmon',title='Trigram Frequency')

#Get sentiment from comments
data[message_col_name] = [str(thing) for thing in data[message_col_name]]
sentiment = []
for i in range(len(data)):
    blob = TextBlob(data[message_col_name][i])
    for sentence in blob.sentences:
        sentiment.append(sentence.sentiment.polarity)
data['sentiment']=sentiment

#Plot
plt.plot(range(0,len(data)),data['sentiment'])
plt.title('Comments Polarity')
plt.xlabel('Comment Number')
plt.ylabel('Polarity')
plt.show()
data.to_csv(r'.\IndiaTodayYTRheaChak_Replies_Sentiment.csv',header=True)