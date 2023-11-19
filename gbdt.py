import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve,auc
from sklearn import metrics
from sklearn.model_selection import GridSearchCV 
from sklearn.preprocessing import LabelEncoder

#读取数据
data_model = pd.read_csv(r'new-train.csv')
#将数据集中的字符串转化为代表类别的数字。因为sklearn的决策树只识别数字
le = LabelEncoder()
for col in data_model.columns:    
    data_model[col] = le.fit_transform(data_model[col].astype(str))
#划分数据集（3、7划分）
y = data_model['Married']
x = data_model.drop('Married', axis=1)
x_train, x_test, y_train, y_test = train_test_split(x, y,random_state=0,train_size=0.9)
#标准化数据
ss_x = StandardScaler()
ss_y = StandardScaler()
x_train = ss_x.fit_transform(x_train)
x_test = ss_x.transform(x_test)

model_GBDT = GradientBoostingClassifier(random_state=42)
model_GBDT.fit(x_train,y_train)
y_pred = model_GBDT.predict(x_test)
y_predprob = model_GBDT.predict_proba(x_train)[:,1]
print ("Accuracy : %.4g" % metrics.accuracy_score(y_test, y_pred))

