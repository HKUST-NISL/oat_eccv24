import random
import pickle
import numpy as np
file='dataset/processdata/dataset_amazon'
with open(file, "rb") as fp:  # Unpickling
    raw_data = pickle.load(fp)

data = raw_data

# raw_data = torch.load(file)
data_length = len(data)
# Generate the list from 453 to 891
number_list = list(range(0, data_length))

# Shuffle the list randomly
random.shuffle(number_list)
#indices = data[np.array(number_list)]

with open('dataset/processdata/splitlist_all_amazon.txt', 'w') as f:
    f.write('\n'.join(map(str, number_list)))




