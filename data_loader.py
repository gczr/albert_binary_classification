
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


# 读取txt文件
def read_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = [line.strip() for line in f.readlines()]

    labels, texts = [], []
    for line in content:
        parts = line.split()
        label, text = parts[0], ''.join(parts[1:])
        labels.append(label)
        texts.append(text)

    return labels, texts

