from sys import argv
from glob import glob
assert len(argv)>=3,"Format python max_ensemble.py output_file_name <list of patterns for input filenames>"
files = argv[2:]
print 'Input Files:',files
import os
assert not os.path.exists(argv[1]),'output file %s already exists'%argv[1]
outfile = file(argv[1],'w')
from collections import defaultdict
file_readers = []
for filename in files:
	file_readers.append(open(filename))
for f in file_readers:
	line0 = f.readline().strip()
print>>outfile,line0

for it in range(700640):
	if(it%25000==0):
		print it
	current_preds = defaultdict(lambda:0)
	video_id = None
	for f in file_readers:
		line = f.readline()
		line = line.split(',')
		if video_id is None:
			video_id = line[0]
		else:
			assert video_id == line[0],"index mismatch at %d in file %d"%(it,files[file_readers.index(f)])
		preds = line[1].split()
		for j in range(0,len(preds),2):
			label = preds[j]
			prob = float(preds[j+1])
			current_preds[label]+=prob

	preds_tuples = [(current_preds[x]/len(files),x) for x in current_preds]
	preds_tuples.sort()
	preds_tuples = preds_tuples[::-1]
	preds_tuples = preds_tuples[:50]
	# preds_tuples = filter(lambda x: x[0]>0.85,preds_tuples)
	preds_tuples_string = ' '.join(['%s %0.6f'%(b,a) for a,b in preds_tuples])
	print>>outfile,video_id+','+preds_tuples_string

outfile.close()
