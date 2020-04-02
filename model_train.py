
# coding: utf-8

# In[1]:


import os
import numpy as np
import pandas as pd
from data_loader import read_data
from keras.utils import to_categorical
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import Input, BatchNormalization, Dense
import matplotlib.pyplot as plt

from albert_zh.extract_feature import BertVector


# #### 读取训练样本数据和测试样本数据

# In[2]:


train_path,test_path = 'data/train.txt','data/test.txt'

labels, texts = read_data(train_path)
df_train = pd.DataFrame({'label': labels, 'text': texts})

labels, texts = read_data(test_path)
df_test = pd.DataFrame({'label': labels, 'text': texts})

df_train['text_len'] = df_train['text'].apply(lambda x: len(x))
#也就是在75%分位的时候，句子长度为102，所以往albert里面灌入的时候，max_seq_len设置为100即可
df_train.describe()


# #### 模型数据准备

# In[3]:


# 直接从albert中价值预训练好的模型
bert_model = BertVector(pooling_strategy="REDUCE_MEAN", max_seq_len=100)

print('begin encoding')
#利用albert模型对汉字进行编码，生成词向量,这里面的[0]就代表只取第一位的cls的位置的embedding输出，albert的词向量长度是312
word_endoder = lambda text: bert_model.encode([text])["encodes"][0]
df_train['x'] = df_train['text'].apply(word_endoder)
df_test['x']  = df_test['text'].apply(word_endoder)
print('end encoding')
print(df_train.head())
x_train = np.array([vec for vec in df_train['x']])
x_test = np.array([vec for vec in df_test['x']])
y_train = np.array([vec for vec in df_train['label']])
y_test = np.array([vec for vec in df_test['label']])
print('x_train: ', x_train.shape)

# Convert class vectors to binary class matrices.
num_classes = 2
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)


# #### 构建keras模型

# In[4]:


# 创建模型
x_in = Input(shape=(312, ))
x_out = Dense(32, activation="relu")(x_in)
x_out = BatchNormalization()(x_out)
x_out = Dense(num_classes, activation="softmax")(x_out)
model = Model(inputs=x_in, outputs=x_out)
print(model.summary())

model.compile(loss='categorical_crossentropy',
              optimizer=Adam(),
              metrics=['accuracy'])

# 模型训练以及评估
history = model.fit(x_train, y_train, validation_data=(x_test, y_test), batch_size=8, epochs=20)
#模型存储
model.save('visit_classify.h5')
print(model.evaluate(x_test, y_test))


# In[5]:


# 绘制loss和acc图像
plt.subplot(2, 1, 1)
epochs = len(history.history['loss'])
plt.plot(range(epochs), history.history['loss'], label='loss')
plt.plot(range(epochs), history.history['val_loss'], label='val_loss')
plt.legend()

plt.subplot(2, 1, 2)
epochs = len(history.history['accuracy'])
plt.plot(range(epochs), history.history['accuracy'], label='acc')
plt.plot(range(epochs), history.history['val_accuracy'], label='val_acc')
plt.legend()
plt.savefig("loss_acc.png")
plt.show()

