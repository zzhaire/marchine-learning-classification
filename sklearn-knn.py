import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
data = pd.read_csv('new2.csv')
data.head()

# X = data.iloc[:,0:16]
# Y = data.iloc[:,16]


# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)


# from sklearn.neighbors import KNeighborsClassifier,RadiusNeighborsClassifier
# n1=15
# n2=22
# n3=2.28
# model1 =KNeighborsClassifier(n_neighbors=n1)
# model1.fit(X_train, Y_train)
# score1 = model1.score(X_test,Y_test)
# model2 =KNeighborsClassifier(n_neighbors=n2, weights='distance')
# model2.fit(X_train, Y_train)
# score2 = model2.score(X_test, Y_test)

# model3 =RadiusNeighborsClassifier(radius=n3)
# model3.fit(X_train, Y_train)
# score3 = model3.score(X_test, Y_test)

# print(score1, score2, score3)
# print("n1:",n1,"  n2:",n2,"  n3:",n3)






import pandas as pd
import numpy as np
from collections import Counter
from sklearn.metrics import accuracy_score

class KNNClassifier:
    def __init__(self, k=3):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        predictions = []
        for i in range(len(X_test)):
            distances = np.linalg.norm(self.X_train - X_test.iloc[i], axis=1)
            indices = np.argsort(distances)[:self.k]
            k_nearest_labels = self.y_train.iloc[indices]
            most_common = Counter(k_nearest_labels).most_common(1)
            prediction = most_common[0][0]
            predictions.append(prediction)
        return pd.Series(predictions, index=X_test.index)


# 读取CSV文件
df = pd.read_csv('new3.csv')
# df = pd.read_csv('new-train.csv')


# 假设数据集中最后一列是目标变量，其他列是特征
X = df.iloc[:, :-1]  # 特征
y = df.iloc[:, -1]   # 目标变量

# 将数据集划分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# 初始化自己的KNN分类器，设置K值（这里假设K=3）
knn_classifier = KNNClassifier(k=70)

# 在训练集上训练KNN分类器
knn_classifier.fit(X_train, y_train)

# 在测试集上进行预测
y_pred = knn_classifier.predict(X_test)
# print(y_pred)

# 计算分类准确率
accuracy = accuracy_score(y_test, y_pred)
print(f'准确率：{accuracy:.2f}')