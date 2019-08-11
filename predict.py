#!/usr/bin/env python
# coding: utf-8
"""
@File     :predict.py
@Copyright: zhiyou720
@Contact  : zhiyou720@gmail.com
@Date     :2019/8/11
@Desc     : 
"""
import keras
from train import x_test, y_test
from model import Attention

model = keras.models.load_model('res.model')
# model.load_weights('./config.pkl')

score = model.evaluate(x_test, y_test, batch_size=16)
print(score)
print('Test...')
# x_test = x_test[:10]
# y_test = y_test[:10]
result = model.predict(x_test)
print(result)

# for item in x_test:
#     result = model.predict([item])
#     print(result)
#     print()

# print(x_test)
def bubbleSort(arr):
    n = len(arr)

    # 遍历所有数组元素
    for i in range(n):

        # Last i elements are already in place
        for j in range(0, n - i - 1):

            if arr[j] < arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]


res = []
for item in result:
    item = list(item)
    tmp = item[:]
    bubbleSort(tmp)
    __ = []
    __.append(item.index(tmp[0]))
    __.append(item.index(tmp[1]))
    __.append(item.index(tmp[4]))
    res.append(__)

print(res)

y_true = []

for y in y_test:
    _ = list(y).index(1)
    y_true.append(_)
    print(_)

score = 0
total = len(x_test)

for i in range(len(y_true)):
    print(res[i], y_true[i])

    if y_true[i] in res[i]:
        score += 1
print(score)
print(total)
