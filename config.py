from data_helper import DataLoader
from sklearn.model_selection import train_test_split

train = True  # 是否训练模型
GPU = True  # 是否开启GPU模式
VALIDATION = False  # 不用改
USE_ALL_DATA = False  # 是否使用全部训练数据 做未来预测一定要选True
SHUFFLE = False  # 打乱数据顺序，我们数据本来就是有序的，不建议打乱
DIY_FEATURE = False  # 是否自行添加特征了, 如果需要添加，请直接在原始数据后增加新的列
PLOT_HEAT_MAP = False  # 是否在读取完数据后绘制数据点分布热力图

BASE_FEATURE_FACTOR_NUM = 5  # 基础特征的数量 5个 不能改 如果需要改的话要改代码
DIY_FEATURE_FACTOR_NUM = 0  # 自定义特征的数量 有几列填几

TOTAL_FEATURE_FACTOR_NUM = BASE_FEATURE_FACTOR_NUM + DIY_FEATURE_FACTOR_NUM

TIME_POINT_CYCLE = 288  # 时间点循环
DATA_TO_PREDICT = 'A'  # 选择 A B DIFF DIFF_ABS 填入, 可以控制模型预测A还是B还是差值还是差值的绝对值

batch_size = 128
embedding_dims = 200
epochs = 17

max_len = 10  # 3 is best
class_num = 8  # n分类
ONE_CLASS_SET = [0, 8, 9]  # 把几个数字视为一类, 最小的数字放在最前面
train_data_path = './data/stage_2_data.csv'
MODEL_PATH = './model/res.stage2.model'

print('Loading data...')

if train:
    data = DataLoader(train_data_path, time_point_cycle=TIME_POINT_CYCLE, diy_feature=False,
                      one_class_set=ONE_CLASS_SET, plot_heat_map=PLOT_HEAT_MAP, train=True,
                      seq_len=max_len, class_num=class_num)
else:
    data = DataLoader(train_data_path, time_point_cycle=TIME_POINT_CYCLE, diy_feature=False,
                      one_class_set=ONE_CLASS_SET, plot_heat_map=PLOT_HEAT_MAP, train=False,
                      seq_len=max_len, class_num=class_num)

x, y, vocab = data.x_train, data.y_train, data.vocabulary

if USE_ALL_DATA:
    train_set_x = x
    train_set_y = y
else:
    train_set_x = x[:-288]
    train_set_y = y[:-288]

test_set_x = x[-288:]
test_set_y = y[-288:]

x_train, x_valid, y_train, y_valid = train_test_split(train_set_x, train_set_y, test_size=0.01, shuffle=SHUFFLE,
                                                      random_state=123)

if VALIDATION:
    train_set_x = x_train
    train_set_y = y_train

print(len(x_train), 'train sequences')
print(len(x_valid), 'test sequences')
