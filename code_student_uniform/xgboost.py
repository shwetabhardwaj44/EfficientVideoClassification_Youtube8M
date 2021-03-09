import numpy as np, scipy as sp, sklearn as sk, pandas as pd, time, sklearn.ensemble, sys, xgboost as xgb
from multiprocessing import Pool

# class indices
class_begin = int(sys.argv[1])
class_end = min(class_begin+64,4716)

# train data
train_data = './csv_data/train_data.csv'

train_X = np.array(pd.read_csv(train_data, header=None))
train_size = len(train_X)
print 'train data read'

# train labels
train_labels = './csv_data/train_labels.csv'
labels_reader = open(train_labels)

data_Y = [map(int,x.strip().split(',')) for x in labels_reader.readlines()]
train_Y = np.zeros([train_size,4716],dtype=np.int32)

for i,j in enumerate(data_Y):
	train_Y[i,j] = 1

# test data
test_data = './csv_data/test_data.csv'

test_X = np.array(pd.read_csv(test_data, header=None))
print 'test data read'

# run for class: class_idx
def run(class_idx):
	beg=time.time()
	c = xgb.XGBClassifier(n_jobs=1,max_depth=4)
	c.fit(train_X,train_Y[:,class_idx])
	print time.time()-beg, 'for class %d'%class_idx

	beg=time.time()
	ret = c.predict_proba(test_X)
	
	f = file('predictions_%04d'%class_idx,'w')
	for i in ret[:,1]:
		print>>f,'%0.6f'%i
	f.close()
	print time.time()-beg, 'for class %d predictions'%class_idx

# parallelize over 64 cores
p = Pool(64)
ret = p.map(run,range(class_begin,class_end))