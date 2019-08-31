from data_helper import DataLoader
from sklearn.model_selection import train_test_split

train = True  # 是否训练模型
GPU = True  # 是否开启GPU模式
VALIDATION = False  # 不用改
USE_ALL_DATA = True  # 是否使用全部训练数据 做未来预测一定要选True

batch_size = 32
embedding_dims = 200
epochs = 17

max_len = 10
class_num = 10
train_data_path = './data/new_train_a+week.csv'
MODEL_PATH = './model/res.model'

print('Loading data...')

if train:
    data = DataLoader(train_data_path, train=True, seq_len=max_len, class_num=class_num)
else:
    data = DataLoader(train_data_path, train=False, seq_len=max_len, class_num=class_num)
x, y, vocab = data.x_train, data.y_train, data.vocabulary

if USE_ALL_DATA:
    train_set_x = x
    train_set_y = y
else:
    train_set_x = x[:-288]
    train_set_y = y[:-288]

test_set_x = x[-288:]
test_set_y = y[-288:]

x_train, x_valid, y_train, y_valid = train_test_split(train_set_x, train_set_y, test_size=0.01, shuffle=False,
                                                      random_state=123)

if VALIDATION:
    train_set_x = x_train
    train_set_y = y_train

print(len(x_train), 'train sequences')
print(len(x_valid), 'test sequences')
