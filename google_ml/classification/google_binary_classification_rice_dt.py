# -*- coding: utf-8 -*-
"""
使用决策树对大米品种（Cammeo vs. Osmancik）进行二元分类。
"""

# @title 导入所需库
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, roc_auc_score,
    classification_report, confusion_matrix
)
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns # 用于更美观的混淆矩阵可视化
import os # 导入 os 模块检查文件

# Configure matplotlib to use a font that supports Chinese characters
# Try 'SimHei' first, commonly available on Windows
try:
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False # Ensure minus signs are displayed correctly
    print("\n配置 Matplotlib 使用 SimHei 字体。")
except Exception as e:
    print(f"\n警告：设置字体时出错 ({e})。请确保系统中安装了支持中文的字体（如 SimHei, Microsoft YaHei）。")
    print("绘图中的中文可能无法正常显示。")

print("成功导入所需库。")

# @title 加载数据集
# 定义本地文件名和下载 URL
local_csv_file = "Rice_Cammeo_Osmancik.csv"
csv_url = "https://download.mlcc.google.com/mledu-datasets/Rice_Cammeo_Osmancik.csv"

# 检查本地文件是否存在
if os.path.exists(local_csv_file):
    try:
        rice_dataset_raw = pd.read_csv(local_csv_file)
        print(f"成功从本地文件 {local_csv_file} 加载数据集。")
    except Exception as e:
        print(f"从本地文件 {local_csv_file} 加载数据时出错: {e}")
        print("请检查文件是否损坏或权限问题。")
        exit()
else:
    print(f"本地未找到 {local_csv_file}，尝试从 URL 下载...")
    try:
        rice_dataset_raw = pd.read_csv(csv_url)
        print(f"成功从 {csv_url} 下载数据集。")
        # 可选：将下载的数据保存到本地，方便下次运行
        try:
            rice_dataset_raw.to_csv(local_csv_file, index=False)
            print(f"数据集已保存到本地 {local_csv_file}")
        except Exception as e:
            print(f"保存数据集到本地失败: {e}")
    except Exception as e:
        print(f"从 URL 下载数据集失败: {e}")
        print(f"请确保 {local_csv_file} 文件存在于工作目录，或检查网络连接和 URL 是否有效。")
        exit() # 如果数据加载失败，则退出脚本

# @title 数据准备
# 选择特征和目标变量
# 使用原始神经网络代码中选择的以及其他所有数值特征
# features = [
#     'Area', 'Perimeter', 'Major_Axis_Length', 'Minor_Axis_Length',
#     'Eccentricity', 'Convex_Area', 'Extent'
# ]
# 与模型google_binary_classification_rice.py中选择的特征一致
features = [
    'Area', 'Major_Axis_Length', 'Eccentricity'
]
target = 'Class' # 目标列名

# 检查特征列和目标列是否存在于 DataFrame 中
missing_features = [f for f in features if f not in rice_dataset_raw.columns]
if missing_features:
    print(f"错误：数据集中缺少以下特征列: {missing_features}")
    exit()
if target not in rice_dataset_raw.columns:
    print(f"错误：数据集中缺少目标列: {target}")
    exit()

X = rice_dataset_raw[features]
y_raw = rice_dataset_raw[target]

# 将类别标签 ('Cammeo', 'Osmancik') 转换为数值 (0 和 1)
# 使用 LabelEncoder 可以自动处理，它会按字母顺序分配标签（例如，Cammeo -> 0, Osmancik -> 1）
le = LabelEncoder()
y = le.fit_transform(y_raw)

# 打印标签编码的映射关系，了解哪个数字代表哪个类别
print("\n标签编码映射:")
class_mapping = {i: class_name for i, class_name in enumerate(le.classes_)}
print(class_mapping)
# 这将在后续的报告和可视化中用到

# @title 数据划分
# 将数据集划分为训练集 (80%) 和测试集 (20%)
# random_state 确保每次划分结果一致，便于复现
# stratify=y 确保训练集和测试集中的类别比例与原始数据集大致相同，这对于不平衡数据集尤其重要
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n原始数据集大小: {len(rice_dataset_raw)} 样本")
print(f"训练集大小: {X_train.shape[0]} 样本 ({len(X_train)/len(rice_dataset_raw):.0%})")
print(f"测试集大小: {X_test.shape[0]} 样本 ({len(X_test)/len(rice_dataset_raw):.0%})")

# @title 创建、训练和评估决策树模型

# 1. 创建决策树分类器实例
# 可以调整超参数以优化性能或控制复杂度，防止过拟合
# 例如：尝试不同的 max_depth, min_samples_split, min_samples_leaf, criterion ('gini' or 'entropy')
dt_classifier = DecisionTreeClassifier(
    max_depth=6,          # 限制树的最大深度为 6 (可调)
    min_samples_split=20, # 节点至少包含 20 个样本才能分裂 (可调)
    min_samples_leaf=10,  # 叶节点至少包含 10 个样本 (可调)
    random_state=42,      # 保证结果可复现
    criterion='gini'      # 信息增益分裂标准，也可尝试 'entropy'
    # criterion='entropy' # 使用熵作为分裂标准
)

# 2. 训练模型
print("\n开始训练决策树模型...")
dt_classifier.fit(X_train, y_train)
print("模型训练完成。")

# 3. 在测试集上进行预测
y_pred = dt_classifier.predict(X_test)
# 获取每个类别的预测概率，用于计算 AUC
# predict_proba 返回一个数组，每行两个值，代表 P(class=0) 和 P(class=1)
# 我们通常关心正类（这里假设是类别 1）的概率
y_pred_proba = dt_classifier.predict_proba(X_test)[:, 1]

# 4. 评估模型性能
print("\n模型在测试集上的评估结果:")
accuracy = accuracy_score(y_test, y_pred)
# precision, recall 默认计算正类（标签为 1）的指标
# average='weighted' 可以计算考虑了类别样本数量的加权平均指标，更全面反映整体性能
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
auc = roc_auc_score(y_test, y_pred_proba) # AUC 通常针对二分类的正负类概率

print(f"  准确率 (Accuracy): {accuracy:.4f}")
print(f"  加权精确率 (Weighted Precision): {precision:.4f}")
print(f"  加权召回率 (Weighted Recall): {recall:.4f}")
print(f"  AUC (Area Under ROC Curve): {auc:.4f}")

# 5. 打印详细分类报告
print("\n详细分类报告:")
# target_names 应与 LabelEncoder 的 classes_ 顺序一致
print(classification_report(y_test, y_pred, target_names=le.classes_))

# 6. 计算并可视化混淆矩阵
print("\n混淆矩阵:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# 使用 seaborn 绘制更美观的混淆矩阵热力图
try:
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=le.classes_, yticklabels=le.classes_)
    plt.xlabel('预测标签 (Predicted Label)')
    plt.ylabel('真实标签 (True Label)')
    plt.title('决策树模型混淆矩阵')
    plt.tight_layout() # 调整布局防止标签重叠
    plt.show() # 显示混淆矩阵图
except Exception as e:
    print(f"绘制混淆矩阵时出错: {e}")
    print("请确保 matplotlib 和 seaborn 已正确安装。")


# @title 可视化决策树 (可选)
# 决策树的可视化可以帮助理解模型的决策过程
# 如果树很深，可视化可能会变得复杂和难以阅读
print(f"\n尝试可视化训练好的决策树 (max_depth={dt_classifier.max_depth})...")
try:
    # Increase figure height significantly and slightly decrease font size
    plt.figure(figsize=(30, 25)) # Increased height from 15 to 25, width from 25 to 30
    plot_tree(
        dt_classifier,
        feature_names=features,       # 特征名称
        class_names=le.classes_,      # 类别名称 (来自 LabelEncoder)
        filled=True,                  # 使用颜色填充节点以表示类别倾向
        rounded=True,                 # 节点框使用圆角
        proportion=False,             # 显示样本数量而非比例
        impurity=True,                # 显示节点的基尼不纯度/熵
        fontsize=8                    # Decreased font size from 10 to 8
    )
    plt.title(f"决策树可视化 (max_depth={dt_classifier.max_depth})", fontsize=14) # Adjusted title fontsize
    # 可以取消注释下一行来保存图像到文件
    # tree_image_filename = "decision_tree_rice_visualization.png"
    # plt.savefig(tree_image_filename, dpi=300, bbox_inches='tight')
    # print(f"决策树可视化图像已保存为 {tree_image_filename}")
    plt.show() # 显示决策树图
    print("决策树可视化完成。")
except Exception as e:
    print(f"绘制决策树时出错: {e}")
    # 提示：在某些系统上，plot_tree 可能需要安装 graphviz 库及其依赖
    print("可视化可能需要 Graphviz 库，请确保已安装，或者尝试更小的 max_depth。")


print("\n脚本执行完毕。")