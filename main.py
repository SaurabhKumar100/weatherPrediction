import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as mso
from scipy import stats
from scipy.stats import ttest_ind
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.model_selection import train_test_split 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


data=pd.read_csv("seattle-weather.csv")
 
data.shape

import warnings
warnings.filterwarnings('ignore')
sns.countplot("weather",data=data,palette="hls")

countrain=len(data[data.weather=="rain"])
countsun=len(data[data.weather=="sun"])
countdrizzle=len(data[data.weather=="drizzle"])
countsnow=len(data[data.weather=="snow"])
countfog=len(data[data.weather=="fog"])


data.isna().sum()


plt.figure(figsize=(12,6))
axz=plt.subplot(1,2,2)
mso.bar(data.drop(["date"],axis=1),ax=axz,fontsize=12);



df=data.drop(["date"],axis=1)

Q1=df.quantile(0.25)
Q3=df.quantile(0.75)
IQR=Q3-Q1
df=df[~((df<(Q1-1.5*IQR))|(df>(Q3+1.5*IQR))).any(axis=1)]

# df.precipitation=np.sqrt(df.precipitation)
# df.wind=np.sqrt(df.wind)

sns.set(style="darkgrid")
fig,axs=plt.subplots(2,2,figsize=(10,8))
sns.histplot(data=df,x="precipitation",kde=True,ax=axs[0,0],color='green')
sns.histplot(data=df,x="temp_max",kde=True,ax=axs[0,1],color='red')
sns.histplot(data=df,x="temp_min",kde=True,ax=axs[1,0],color='skyblue')
sns.histplot(data=df,x="wind",kde=True,ax=axs[1,1],color='orange')

lc=LabelEncoder()
df["weather"]=lc.fit_transform(df["weather"])


x=((df.loc[:,df.columns!="weather"]).astype(int)).values[:,0:]
y=df["weather"].values

print(df.weather.unique())

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.1,random_state=2)

y_train = y_train.reshape(len(y_train), 1)

from sklearn.preprocessing import StandardScaler 
sc1 = StandardScaler()
sc2 = StandardScaler()
x_train = sc1.fit_transform(x_train)
y_train = sc2.fit_transform(y_train)

knn=KNeighborsClassifier(n_neighbors=5, weights='uniform', algorithm='auto', p=2, leaf_size=30, metric='minkowski')
knn.fit(x,np.ravel(y,order="c"))
print("KNN Accuracy:{:.2f}%".format(knn.score(x_test,y_test)*100))



import pickle
import requests

pickle.dump(knn, open('model.pkl','wb'))

model = pickle.load(open('model.pkl','rb'))