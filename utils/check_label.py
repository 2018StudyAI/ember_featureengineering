import pandas as pd
import os
import tqdm

predpath = '/home/choi/Downloads/TestSet_final1_VX' #predict direcotry
labelpath = '~/Downloads/TestSet_final1_VX.csv' #answer

data = pd.read_csv(labelpath, names=['hash', 'y'])

cnt = 0
for _name in tqdm.tqdm(os.listdir(predpath)):
    r = data[data.hash==_name].values
    
    if r is not None:
        cnt+=1

print(cnt)


