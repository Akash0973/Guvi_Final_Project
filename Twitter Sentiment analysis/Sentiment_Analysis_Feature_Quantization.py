import pandas as pd
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import time
from sklearn.preprocessing import MinMaxScaler

start_time = time.time()

data=pd.read_csv('Twitter Sentiment analysis/Cleaned_Data.csv')

tagged_text=[TaggedDocument(words=sentence.split(), tags=[str(i)]) for i, sentence in enumerate(data['Text'])]
model = Doc2Vec(tagged_text, vector_size=20, window=3, min_count=1, workers=4)
embedding=[model.infer_vector(sentence.split()) for sentence in data['Text']]

data=pd.concat(
    [pd.DataFrame(embedding,columns=['Text'+str(i) for i in range(20)]),data],
    axis=1)

data=data.drop('Text',axis=1)

week_map={'Mon':1,'Tue':2,'Wed':3,'Thu':4,'Fri':5,'Sat':6,'Sun':7}
month_map={'Apr':4,'May':5,'Jun':6}

data['Weekday']=data['Weekday'].apply(lambda x: week_map[x])
data['Month']=data['Month'].apply(lambda x: month_map[x])

scaler = MinMaxScaler()

scaled_data = scaler.fit_transform(data)
data = pd.DataFrame(scaled_data, columns=data.columns)

data.to_csv('Features_Target.csv',index=False)

end_time = time.time()
execution_time = end_time - start_time
print(f"Execution time: {execution_time} seconds")