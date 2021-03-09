from scipy.stats import rankdata
from sys import argv
import os
import numpy as np

filename = argv[1]
outfilename = filename+'.rank'
print 'outfilename:',outfilename
assert not os.path.exists(outfilename), outfilename+' already exists'

f=open(filename,'r').readlines()
print 'File read'
first_line = f[0].strip()
f=f[1:]
f = map(lambda x: x.strip().split(','), f)
f = map(lambda x: [x[0],x[1].split(' ')], f)
matrix = np.array([map(float,x[1][1::2]) for x in f])
ranks = rankdata(matrix).reshape(np.shape(matrix))
ranks = ranks/np.max(ranks)
print 'Rank normalized'

out = file(outfilename,'w')
print>>out,first_line
for i,line in enumerate(f):
	print>>out, '%s,%s'%(line[0],' '.join(['%s %0.6f'%(a,b) for a,b in zip(line[1][0::2],ranks[i,:])]))
out.close()