import numpy as np
import pandas as pd
np.random.seed(1)


LABEL_PATH = 'data/light_labels.csv'
full_labels = pd.read_csv(LABEL_PATH)

print(full_labels.head())


grouped = full_labels.groupby('filename')
print(grouped )

print(grouped.apply(lambda x: len(x)).value_counts() )

gb = full_labels.groupby('filename')
print(gb)

grouped_list = [gb.get_group(x) for x in gb.groups]
print(len(grouped_list))

train_index = np.random.choice(len(grouped_list), size=160, replace=False)
test_index = np.setdiff1d(list(range(200)), train_index)

print(len(train_index), len(test_index))

# take first 200 files
train = pd.concat([grouped_list[i] for i in train_index])
test = pd.concat([grouped_list[i] for i in test_index])
print(len(train), len(test))


train.to_csv('data/train_labels.csv', index=None)
test.to_csv('data/test_labels.csv', index=None)