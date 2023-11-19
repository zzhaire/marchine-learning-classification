import pandas as pd
import numpy as np

#计算信息熵
def cal_information_entropy(data):
    data_label = data.iloc[:,-1]
    label_class =data_label.value_counts() #总共有多少类
    Ent = 0
    for k in label_class.keys():
        p_k = label_class[k]/len(data_label)
        Ent += -p_k*np.log2(p_k)
    return Ent

#计算给定数据属性a的信息增益
def cal_information_gain(data, a):
    Ent = cal_information_entropy(data)
    feature_class = data[a].value_counts() #特征有多少种可能
    gain = 0
    for v in feature_class.keys():
        weight = feature_class[v]/data.shape[0]
        Ent_v = cal_information_entropy(data.loc[data[a] == v])
        gain += weight*Ent_v
    return Ent - gain

#获取标签最多的那一类
def get_most_label(data):
    data_label = data.iloc[:,-1]
    label_sort = data_label.value_counts(sort=True)
    return label_sort.keys()[0]

#挑选最优特征，即信息增益最大的特征
def get_best_feature(data):
    features = data.columns[:-1]
    res = {}
    for a in features:
        temp = cal_information_gain(data, a)
        res[a] = temp
    res = sorted(res.items(),key=lambda x:x[1],reverse=True)
    return res[0][0]

##将数据转化为（属性值：数据）的元组形式返回，并删除之前的特征列
def drop_exist_feature(data, best_feature):
    attr = pd.unique(data[best_feature])
    new_data = [(nd, data[data[best_feature] == nd]) for nd in attr]
    new_data = [(n[0], n[1].drop([best_feature], axis=1)) for n in new_data]
    return new_data

#创建决策树
# def create_tree(data):
#     data_label = data.iloc[:,-1]
#     if len(data_label.value_counts()) == 1: #只有一类
#         return data_label.values[0]
#     if all(len(data[i].value_counts()) == 1 for i in data.iloc[:,:-1].columns): #所有数据的特征值一样，选样本最多的类作为分类结果
#         return get_most_label(data)
#     best_feature = get_best_feature(data) #根据信息增益得到的最优划分特征
#     Tree = {best_feature:{}} #用字典形式存储决策树
#     exist_vals = pd.unique(data[best_feature]) #当前数据下最佳特征的取值
#     if len(exist_vals) != len(column_count[best_feature]): #如果特征的取值相比于原来的少了
#         no_exist_attr = set(column_count[best_feature]) - set(exist_vals) #少的那些特征
#         for no_feat in no_exist_attr:
#             Tree[best_feature][no_feat] = get_most_label(data) #缺失的特征分类为当前类别最多的

#     for item in drop_exist_feature(data,best_feature): #根据特征值的不同递归创建决策树
#         Tree[best_feature][item[0]] = create_tree(item[1])
#     return Tree

from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier

def create_tree(data):
    # X 是特征， Y 是标签
    X = data.iloc[:, :-1]
    Y = data.iloc[:, -1]

    # 创建决策树分类器
    tree_classifier = DecisionTreeClassifier(
        criterion='entropy',
        max_depth=8,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features=None,
        random_state=42,  # 添加 random_state 参数以确保结果的可重复性
    )

    # 在训练数据上拟合模型
    tree_classifier.fit(X, Y)

    return tree_classifier



def predict(tree_model, test_data):
    if isinstance(tree_model, DecisionTreeClassifier):
        # 如果是 DecisionTreeClassifier 对象
        predicted_label = tree_model.predict([test_data])[0]
        return predicted_label

    elif isinstance(tree_model, dict):
        # 如果是手动实现的字典型决策树
        first_feature = list(tree_model.keys())[0]
        second_dict = tree_model[first_feature]
        input_first = test_data.get(first_feature)

        if input_first is None:
            return 'unknown'

        input_value = second_dict.get(input_first)

        if isinstance(input_value, dict):
            class_label = predict(input_value, test_data)
        else:
            class_label = input_value

        return class_label

    else:
        raise ValueError("Unsupported tree model type")

def calculate_accuracy(Tree, test_data):
    correct_predictions = 0

    for i in range(len(test_data)):
        predicted_label = predict(Tree, test_data.iloc[i, 0:9])
        actual_label = test_data.iloc[i, 9]

        if predicted_label == actual_label:
            correct_predictions += 1

    accuracy = correct_predictions / len(test_data)
    return accuracy


from sklearn.model_selection import train_test_split
if __name__ == '__main__':
    #读取数据
    data = pd.read_csv('new3.csv')

    #统计每个特征的取值情况作为全局变量
    column_count = dict([(ds, list(pd.unique(data[ds]))) for ds in data.iloc[:, :-1].columns])

    # X = data.iloc[:,0:9]
    # Y = data.iloc[:,9]
    # X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=22)

    train_data, test_data = train_test_split(data, test_size=0.3, random_state=42)
    # #创建决策树
    dicision_Tree = create_tree(train_data)
    # print(dicision_Tree)
    # result = predict(dicision_Tree,Y_test)

    accuracy = calculate_accuracy(dicision_Tree, test_data)
    print(f'准确率: {accuracy * 100:.2f}%')