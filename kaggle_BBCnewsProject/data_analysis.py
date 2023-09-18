import pandas as pd
import matplotlib.pyplot as plt
train_df = pd.read_csv('BBCNews/data/BBC News Train.csv', sep=',', nrows=100)
for index, row in train_df.iterrows():
    content=row['Text']
    label=row['Category']
    print(label)
pd.set_option('display.max_columns', None)
print(train_df.head())
train_df['text_len'] = train_df["Text"].apply(lambda x: len(x.split(' ')))
print(train_df['text_len'].describe())
_ = plt.hist(train_df['text_len'], bins=200) #bins是要划分的区间数
plt.xlabel('Text char count')
plt.title("Histogram of char count")
plt.show()