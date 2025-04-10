# @title Install required libraries
# Please run: pip install -r requirements.txt
# 这是安装所需依赖的命令，你需要在命令行运行上述命令来安装所有必要的库

# 这个模型基于以下论文：https://ijisae.org/index.php/IJISAE/article/view/1068/599

# @title Load the imports
# 导入所有必要的库和模块

import io  # 处理IO操作
import keras  # 深度学习框架，用于构建和训练神经网络
from matplotlib import pyplot as plt  # 用于数据可视化
from matplotlib.lines import Line2D  # 用于自定义图例
import ml_edu.experiment  # Google的机器学习教育模块
import ml_edu.results  # 用于结果分析和可视化
import numpy as np  # 科学计算库，用于数值运算
import pandas as pd  # 数据分析库，用于数据处理
import plotly.express as px  # 交互式数据可视化

# 设置pandas显示选项
pd.options.display.max_rows = 10  # 限制显示的最大行数
pd.options.display.float_format = "{:.1f}".format  # 设置浮点数格式为一位小数

print("Ran the import statements.")

# @title Load the dataset
# 从URL加载米粒数据集（Cammeo和Osmancik两种类型的米粒）
# rice_dataset_raw = pd.read_csv("https://download.mlcc.google.com/mledu-datasets/Rice_Cammeo_Osmancik.csv")
rice_dataset_raw = pd.read_csv("Rice_Cammeo_Osmancik.csv")

# @title
# 选择我们关心的特征列和目标列
rice_dataset = rice_dataset_raw[[
    'Area',           # 米粒面积
    'Perimeter',      # 米粒周长
    'Major_Axis_Length',  # 长轴长度（米粒的最长直线距离）
    'Minor_Axis_Length',  # 短轴长度（垂直于长轴的最长距离）
    'Eccentricity',   # 离心率（描述米粒形状的椭圆程度）
    'Convex_Area',    # 凸包面积（米粒的凸包所覆盖的面积）
    'Extent',         # 范围（米粒面积与其边界框面积的比率）
    'Class',          # 分类（Cammeo或Osmancik）
]]

# 查看数据集的统计信息
rice_dataset.describe()

# @title Solutions (运行这个单元获取答案)
# 这部分代码用于探索性数据分析，打印关于数据集的一些见解

print(
    f'The shortest grain is {rice_dataset.Major_Axis_Length.min():.1f}px long,'
    f' while the longest is {rice_dataset.Major_Axis_Length.max():.1f}px.'
)
print(
    f'The smallest rice grain has an area of {rice_dataset.Area.min()}px, while'
    f' the largest has an area of {rice_dataset.Area.max()}px.'
)
print(
    'The largest rice grain, with a perimeter of'
    f' {rice_dataset.Perimeter.max():.1f}px, is'
    f' ~{(rice_dataset.Perimeter.max() - rice_dataset.Perimeter.mean())/rice_dataset.Perimeter.std():.1f} standard'
    f' deviations ({rice_dataset.Perimeter.std():.1f}) from the mean'
    f' ({rice_dataset.Perimeter.mean():.1f}px).'
)
print(
    f'This is calculated as: ({rice_dataset.Perimeter.max():.1f} -'
    f' {rice_dataset.Perimeter.mean():.1f})/{rice_dataset.Perimeter.std():.1f} ='
    f' {(rice_dataset.Perimeter.max() - rice_dataset.Perimeter.mean())/rice_dataset.Perimeter.std():.1f}'
)

# 下面的代码被注释掉了，但它可以用来创建特征对之间的散点图，按类别着色
# for x_axis_data, y_axis_data in [
#     ('Area', 'Eccentricity'),
#     ('Convex_Area', 'Perimeter'),
#     ('Major_Axis_Length', 'Minor_Axis_Length'),
#     ('Perimeter', 'Extent'),
#     ('Eccentricity', 'Major_Axis_Length'),
# ]:
#   px.scatter(rice_dataset, x=x_axis_data, y=y_axis_data, color='Class').show()

# 数据标准化：计算Z分数
# 标准化是机器学习中常见的预处理步骤，将特征转换为均值为0，标准差为1
# 目的是使不同量纲的特征可比，加速梯度下降收敛

feature_mean = rice_dataset.mean(numeric_only=True)  # 计算每个特征的均值
feature_std = rice_dataset.std(numeric_only=True)    # 计算每个特征的标准差
numerical_features = rice_dataset.select_dtypes('number').columns  # 选择数值型列
normalized_dataset = (
    rice_dataset[numerical_features] - feature_mean
) / feature_std  # 应用Z分数公式：(x - mean) / std

# 将原始类别标签复制到标准化后的数据集
normalized_dataset['Class'] = rice_dataset['Class']

# 查看标准化后的数据集的前几行
print(normalized_dataset.head())

# 设置随机种子以确保结果可重现
keras.utils.set_random_seed(42)

# 创建一个布尔标签列：Cammeo=1，Osmancik=0
# 这是将文本标签转换为二元分类模型可以使用的数值
normalized_dataset['Class_Bool'] = (
    # 如果类别是Cammeo返回True，如果是Osmancik返回False
    normalized_dataset['Class'] == 'Cammeo'
).astype(int)  # 转换为整数（0和1）
print(normalized_dataset.sample(10))  # 随机显示10行查看结果

# 定义我们要用于训练模型的特征
# 注意我们只选择了部分特征，这是特征选择的一种形式
input_features = [
    'Eccentricity',       # 米粒的椭圆度
    'Major_Axis_Length',  # 米粒的最长轴
    'Area',               # 米粒的面积
]

# 创建80%和90%位置的索引，用于数据集划分
number_samples = len(normalized_dataset)
index_80th = round(number_samples * 0.8)
index_90th = index_80th + round(number_samples * 0.1)

# 随机打乱数据集并分为训练集(80%)、验证集(10%)和测试集(10%)
# 训练集：用于模型学习参数
# 验证集：用于调整超参数，防止过拟合
# 测试集：用于最终评估模型性能，这部分数据模型从未见过
shuffled_dataset = normalized_dataset.sample(frac=1, random_state=100)
train_data = shuffled_dataset.iloc[0:index_80th]
validation_data = shuffled_dataset.iloc[index_80th:index_90th]
test_data = shuffled_dataset.iloc[index_90th:]

# 查看测试集的前5行
test_data.head()

# 定义标签列，以便在特征数据中排除它们
label_columns = ['Class', 'Class_Bool']

# 准备训练、验证和测试数据集的特征与标签
# 特征：输入到模型的数据
# 标签：模型需要预测的目标值
train_features = train_data.drop(columns=label_columns)
train_labels = train_data['Class_Bool'].to_numpy()
validation_features = validation_data.drop(columns=label_columns)
validation_labels = validation_data['Class_Bool'].to_numpy()
test_features = test_data.drop(columns=label_columns)
test_labels = test_data['Class_Bool'].to_numpy()

# @title 定义创建和训练模型的函数


def create_model(
    settings: ml_edu.experiment.ExperimentSettings,  # 实验设置参数
    metrics: list[keras.metrics.Metric],             # 评估指标列表
) -> keras.Model:
  """创建并编译一个简单的分类模型。"""
  # 为每个输入特征创建一个Keras输入层
  model_inputs = [
      keras.Input(name=feature, shape=(1,))
      for feature in settings.input_features
  ]
  # 使用Concatenate层将不同的输入特征连接成一个张量
  # 例如输入为[特征1值, 特征2值, 特征3值]
  
  concatenated_inputs = keras.layers.Concatenate()(model_inputs)
  # 添加一个全连接层（Dense层），单位为1，使用sigmoid激活函数
  # sigmoid将输出压缩到0-1之间，适合二元分类问题
  model_output = keras.layers.Dense(
      units=1, name='dense_layer', activation=keras.activations.sigmoid
  )(concatenated_inputs)
  
  # 创建模型，指定输入和输出
  model = keras.Model(inputs=model_inputs, outputs=model_output)
  
  # 编译模型，指定优化器、损失函数和评估指标
  # RMSprop：自适应学习率优化算法
  # BinaryCrossentropy：二元分类的标准损失函数
  model.compile(
      optimizer=keras.optimizers.RMSprop(
          settings.learning_rate  # 学习率，控制参数更新步长
      ),
      loss=keras.losses.BinaryCrossentropy(),  # 二元交叉熵损失
      metrics=metrics,  # 用于评估模型性能的指标
  )
  return model


def train_model(
    experiment_name: str,  # 实验名称
    model: keras.Model,     # 待训练的模型
    dataset: pd.DataFrame,  # 特征数据集
    labels: np.ndarray,     # 标签
    settings: ml_edu.experiment.ExperimentSettings,  # 实验设置
) -> ml_edu.experiment.Experiment:
  """用数据集训练模型"""

  # 准备每个特征的数据数组，Keras模型的x参数可以是数组列表
  features = {
      feature_name: np.array(dataset[feature_name])
      for feature_name in settings.input_features
  }

  # 训练模型
  # batch_size：每次更新梯度使用的样本数
  # epochs：完整遍历训练集的次数
  history = model.fit(
      x=features,
      y=labels,
      batch_size=settings.batch_size,
      epochs=settings.number_epochs,
  )

  # 创建并返回实验对象，包含实验结果
  return ml_edu.experiment.Experiment(
      name=experiment_name,
      settings=settings,
      model=model,
      epochs=history.epoch,
      metrics_history=pd.DataFrame(history.history),
  )


print('Defined the create_model and train_model functions.')

# 定义第一个实验的设置
settings = ml_edu.experiment.ExperimentSettings(
    learning_rate=0.001,  # 学习率，控制每步参数更新的幅度
    number_epochs=60,     # 训练轮数，遍历整个数据集的次数
    batch_size=100,       # 批大小，每次计算梯度使用的样本数
    classification_threshold=0.5,  # 分类阈值，大于此值预测为正类
    input_features=input_features,  # 输入特征列表
)

# 定义要追踪的评估指标
metrics = [
    # 准确率：正确预测的比例
    keras.metrics.BinaryAccuracy(
        name='accuracy', threshold=settings.classification_threshold
    ),
    # 精确率：预测为正的样本中实际为正的比例
    keras.metrics.Precision(
        name='precision', thresholds=settings.classification_threshold
    ),
    # 召回率：实际为正的样本中被正确预测的比例
    keras.metrics.Recall(
        name='recall', thresholds=settings.classification_threshold
    ),
    # ROC曲线下面积，评估模型在不同阈值下的性能
    keras.metrics.AUC(num_thresholds=100, name='auc'),
]

# 创建模型
model = create_model(settings, metrics)

# 在训练集上训练模型
experiment = train_model(
    'baseline', model, train_features, train_labels, settings
)

# 绘制训练过程中各指标随epoch变化的曲线
# 这可以帮助了解模型的学习进展和是否有过拟合现象
# ml_edu.results.plot_experiment_metrics(experiment, ['accuracy', 'precision', 'recall'])
# ml_edu.results.plot_experiment_metrics(experiment, ['auc'])
ml_edu.results.plot_experiment_metrics(experiment, ['accuracy', 'precision', 'recall', 'auc'])

# 定义比较训练和测试指标的函数
def compare_train_test(experiment: ml_edu.experiment.Experiment, test_metrics: dict[str, float]):
  print('Comparing metrics between train and test:')
  for metric, test_value in test_metrics.items():
    print('------')
    print(f'Train {metric}: {experiment.get_final_metric_value(metric):.4f}')
    print(f'Test {metric}:  {test_value:.4f}')


# 在测试集上评估模型性能
# 训练-测试指标的对比可以帮助判断模型是否过拟合
test_metrics = experiment.evaluate(test_features, test_labels)
compare_train_test(experiment, test_metrics)

# 获取模型的所有权重和偏置
all_weights = experiment.model.get_weights()

# 打印权重和偏置
# 通常，权重矩阵在前，偏置向量在后
# 对于这个简单模型，只有一个 Dense 层有可训练参数
print("Weights and Biases of the Dense layer:")
if len(all_weights) >= 2:
	print("Weights (连接输入特征到输出单元):")
	print(all_weights[0]) # 权重矩阵
	print("\nBias (添加到输出单元):")
	print(all_weights[1]) # 偏置向量
else:
		print("Could not retrieve weights as expected.")
		print(all_weights)
		
# 你也可以查看模型的摘要信息，了解结构和参数数量
print("\nModel Summary:")
experiment.model.summary()