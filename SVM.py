"""
from sklearn.datasets import make_blobs
from sklearn import svm
import numpy as np
import torch
import numpy
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import pandas as pd
import numpy as np
from collections import Counter
from sklearn.metrics import accuracy_score
 
class attDataset(Dataset):
    def __init__(self, path):
        super(attDataset, self).__init__()
        atts = []
        with open(path, 'r', encoding='utf-8') as f:
            # Skip the header which contains column names
            next(f)  # Skip the first line
            for row in f:
                row = row.split(',')
                att = row[:-1]
                att = [float(i) for i in att]  # Convert to float instead of int
                label = int(float(row[-1]))  # Convert the label to int after converting to float
                atts.append([att, label])
            self.atts = atts

    def __getitem__(self, index):
        attribute, label = self.atts[index]
        
        return torch.Tensor(attribute), torch.Tensor([label])  # Return tensors instead of two values
 
    def __len__(self):
        return len(self.atts)

 
 
path_train = r'new-train.csv'
path_test = r'new-train.csv'
 
 
train_features = []
train_label = []
test_features = []
test_label = []
 
train_set =  attDataset(path=path_train)
test_set  =  attDataset(path=path_test)

# 读取CSV文件
df = pd.read_csv('new-train.csv')

# 假设数据集中最后一列是目标变量，其他列是特征
X = df.iloc[:, 0:9] # 特征
Y = df.iloc[:, 9]  # 目标变量
from sklearn.model_selection import train_test_split
# train_set, test_set = train_test_split(df, test_size=0.2, random_state=22)

 
for att,label in train_set:
    train_features.append(att)
    train_label.append(label)
 
for att,label in test_set:
    test_features.append(att)
    test_label.append(label)
 
train_features = numpy.array(train_features)
train_label = numpy.array(train_label)
test_features = numpy.array(test_features)
test_label = numpy.array(test_label)
 
clf = svm.SVC(C=3, gamma=0.05,max_iter=200)
clf.fit(train_features, train_label)
 
 
#Test on Training data
train_result = clf.predict(train_features)
precision = sum(train_result == train_label)/train_label.shape[0]
print('Training precision: ', precision)
 
 
#Test on test data
test_result = clf.predict(test_features)
precision = sum(test_result == test_label)/test_label.shape[0]
print('Test precision: ', precision)




from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 加载数据集
data = pd.read_csv('new-train.csv')

# 从数据集中选择特征列（X）和目标列（y）
feature_columns = ['Gender', 'Age', 'Graduated', 'Profession', 'WorkExperience', 'SpendingScore', 'FamilySize','Segmentation']
target_column = 'Married'

X = data[feature_columns]
y = data[target_column]

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建SVM分类器
clf = SVC(kernel='linear')  # 选择线性核函数

# 在训练集上训练模型
clf.fit(X_train, y_train)

# 在测试集上进行预测
y_pred = clf.predict(X_test)

# 评估模型
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# 打印分类报告
print(classification_report(y_test, y_pred))

# 绘制训练集的散点图
plt.scatter(X_train['Age'], X_train['SpendingScore'], c=y_train, cmap='viridis')

# 绘制决策边界
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()

xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 100), np.linspace(ylim[0], ylim[1], 100))
Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

ax.contour(xx, yy, Z, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])
plt.show()
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 加载数据集
data = pd.read_csv('new-train.csv')

# 从数据集中选择特征列（X）和目标列（y）
feature_columns = ['Gender', 'Age', 'Graduated', 'Profession', 'WorkExperience', 'SpendingScore', 'FamilySize','Segmentation']
target_column = 'Married'

X = data[feature_columns]
y = data[target_column]

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建逻辑回归分类器
logreg = LogisticRegression()

# 在训练集上训练模型
logreg.fit(X_train[['Age', 'SpendingScore']], y_train)

# 在测试集上进行预测
y_pred = logreg.predict(X_test[['Age', 'SpendingScore']])

# 评估模型
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# 打印分类报告
print(classification_report(y_test, y_pred))

# 绘制训练集的散点图
plt.scatter(X_train['Age'], X_train['SpendingScore'], c=y_train, cmap='viridis')

# 绘制决策边界
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()

xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 100), np.linspace(ylim[0], ylim[1], 100))
Z = logreg.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')
plt.show()
"""
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def find_badcase(X, Y):
    bad_list = []
    y = clf.predict(X)
    Y = Y.reset_index(drop=True)  # 重置索引
    for i in range(len(X)):
        if y[i] != Y[i]:
            bad_list.append(i)
    return bad_list

# 加载数据集
data = pd.read_csv('new-train.csv')

# 从数据集中选择特征列（X）和目标列（y）
feature_columns = ['Gender', 'Age', 'Graduated', 'Profession', 'WorkExperience', 'SpendingScore', 'FamilySize','Segmentation']
target_column = 'Married'

X = data[feature_columns]
y = data[target_column]

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建SVM分类器
clf = SVC(kernel='linear')  # 选择线性核函数

# 在训练集上训练模型
clf.fit(X_train, y_train)

bad_idx = find_badcase(X_test,y_test)

n_Support_vector = clf.n_support_  # 支持向量个数
sv_idx = clf.support_  # 支持向量索引
w = clf.coef_  
b = clf.intercept_

# 在测试集上进行预测
y_pred = clf.predict(X_test)

# 评估模型
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# 打印分类报告
print(classification_report(y_test, y_pred))

# 绘制分类平面
ax = plt.subplot(111, projection='3d')
x = np.arange(-1,1,0.01)
y = np.arange(-1,1,0.11)
x, y = np.meshgrid(x, y)
z = (w[0,0]*x + w[0,1]*y + b) / (-w[0,2])
z_mean = np.mean(z)
z_std = np.std(z)
z_standardized = (z - z_mean) / z_std
# 将标准化后的 z 映射到 -1 到 1 的范围
z = 2 * (z_standardized - z_standardized.min()) / (z_standardized.max() - z_standardized.min()) - 1

surf = ax.plot_surface(x, y, z, rstride=1, cstride=1)


# 获取特定的三个特征列作为 X_train
X_train_selected = X_train[['Age', 'Segmentation', 'Profession']]

from sklearn.preprocessing import StandardScaler, LabelEncoder
label_encoder = LabelEncoder()
X_train_selected= StandardScaler().fit_transform(X_train_selected)

# 转换为 NumPy 数组
x_array = np.array(X_train_selected, dtype=float)
y_array = np.array(y_train, dtype=int)

from sklearn.preprocessing import MinMaxScaler
# 创建 MinMaxScaler 对象，指定 feature_range
scaler = MinMaxScaler(feature_range=(-1, 1))
# 对 x_array 进行标准化
x_array = scaler.fit_transform(x_array)

# print(x_array)
# 假设 y_train 中的标签为 0 和 1
pos = x_array[np.where(y_array == 1)]
neg = x_array[np.where(y_array == 0)]
print(pos)
print(neg)

# 创建标准化对象
scaler = StandardScaler()
# 对 pos 数组进行标准化
pos_scaled = pos
neg_scaled = neg
# pos_scaled = scaler.fit_transform(pos)
# neg_scaled = scaler.fit_transform(neg)
# pos_scaled['Age'] = scaler.fit_transform(pos['Age'])
# pos_scaled['Segmentation'] = scaler.fit_transform(pos['Segmentation'])
# pos_scaled['Profession'] = scaler.fit_transform(pos['Profession'])
# # 对 neg 数组进行标准化
# neg_scaled['Age'] = scaler.fit_transform(neg['Age'])
# neg_scaled['Segmentation'] = scaler.fit_transform(neg['Segmentation'])
# neg_scaled['Profession'] = scaler.fit_transform(neg['Profession'])



# 创建 3D 图
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# 绘制三维散点图
ax.scatter(pos_scaled[:, 0], pos_scaled[:, 1], pos_scaled[:, 2], c='r', label='pos')
ax.scatter(neg_scaled[:, 0], neg_scaled[:, 1], neg_scaled[:, 2], c='b', label='neg')

# 设置坐标轴标签
ax.set_xlabel('Age')
ax.set_ylabel('Segmentation')
ax.set_zlabel('Profession')

plt.legend(loc='best')
plt.show()

