import numpy as np
from numpy.core.numeric import NaN
from numpy.lib.function_base import vectorize
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer

message_col_name='snippet.topLevelComment.snippet.textDisplay'

TimesNowYT=pd.read_csv(r'..\CSV\TimesNowYT_Replies.csv',header=0,dtype={message_col_name:str})
IndiaTodayYT=pd.read_csv(r'..\CSV\IndiaTodayYT_Replies.csv',header=0,dtype={message_col_name:str})
NDTVYTEnglish=pd.read_csv(r'..\CSV\NDTVYTEnglish_Replies.csv',header=0,dtype={message_col_name:str})
NDTVYTHindi=pd.read_csv(r'..\CSV\NDTVYTHindi_Replies.csv',header=0,dtype={message_col_name:str})

TimesNowYT=TimesNowYT[[message_col_name]]
IndiaTodayYT=IndiaTodayYT[[message_col_name]]
NDTVYTEnglish=NDTVYTEnglish[[message_col_name]]
NDTVYTHindi=NDTVYTHindi[[message_col_name]]

TimesNowYT.dropna(inplace=True)
IndiaTodayYT.dropna(inplace=True)
NDTVYTEnglish.dropna(inplace=True)
NDTVYTHindi.dropna(inplace=True)

TimesNowYT['Media']=['TimesNowYT' for i in range(len(TimesNowYT))]
IndiaTodayYT['Media']=['IndiaTodayYT' for i in range(len(IndiaTodayYT))]
NDTVYTEnglish['Media']=['NDTVYTEnglish' for i in range(len(NDTVYTEnglish))]
NDTVYTHindi['Media']=['NDTVYTHindi' for i in range(len(NDTVYTHindi))]

TimesNowYT=TimesNowYT.groupby('Media')[message_col_name].apply(' '.join).reset_index()
IndiaTodayYT=IndiaTodayYT.groupby('Media')[message_col_name].apply(' '.join).reset_index()
NDTVYTEnglish=NDTVYTEnglish.groupby('Media')[message_col_name].apply(' '.join).reset_index()
NDTVYTHindi=NDTVYTHindi.groupby('Media')[message_col_name].apply(' '.join).reset_index()

data= [TimesNowYT,IndiaTodayYT,NDTVYTEnglish,NDTVYTHindi]
data=pd.concat(data)

print(len(data))

vectorizer=CountVectorizer()
vec_data=vectorizer.fit_transform(data[message_col_name])

search_params={'n_components': [3,5,10] }
lda=LatentDirichletAllocation(learning_method='online',n_jobs=-1,learning_decay= 0.9,learning_offset=15)
model=GridSearchCV(cv=3,estimator=lda,param_grid=search_params)
model.fit(vec_data)
g_model=model.best_estimator_
g_output=g_model.fit_transform(vec_data)

ll=g_model.score(vec_data)
print("Model score: ",ll)

print("Best estimator: \n",g_model)

MediaTopic=pd.DataFrame(data=np.round(g_output,2), index=data['Media'], columns=[i for i in range(g_model.n_components)])

MediaTopic.to_csv(r'.\MediaTopic.csv')