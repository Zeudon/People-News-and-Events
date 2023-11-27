from ibm_watson import ToneAnalyzerV3
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator, authenticator
import pandas as pd

# Authenticator
apikey = 'rz61TqdUDMkHTJbbA-imq-XURpZrQvh4uc9qjdcXlmUB'
url = 'https://api.au-syd.tone-analyzer.watson.cloud.ibm.com/instances/d6063dd5-05b3-4c13-a447-8afb1ed8802e'
authenticator=IAMAuthenticator(apikey)
ta=ToneAnalyzerV3(version='2017-09-21', authenticator=authenticator)
ta.set_service_url(url)

# YouTube Columns
# Top Level Comments
# message_col_name='snippet.topLevelComment.snippet.textDisplay'
message_col_name='Comment'
# Replies
# message_col_name='snippet.textOriginal'

# Facebook Columns
# message_col_name='message'

data=pd.read_csv(r"..\Stan Swamy\Data\YouTube\NDTVYT.csv",header=0)
# data=data[['id','parent_id',message_col_name,'level']]
# data=data[data['level']==1]
data=data[['Name',message_col_name]]
data=data[data[message_col_name]!='']
# data=data[['id',message_col_name]]
data.reset_index(inplace=True,drop=True)
res=[]
# print(data.head())

for i in range(len(data)):
    msg=str(data[message_col_name][i])
    temp_res = ta.tone(msg).get_result()
    res.append(temp_res)

data['tone']=res
data.to_csv(r"..\Stan Swamy\Tone\YouTube\Top Level Comments\NDTVYT_Tone.csv",header=True,index=False)