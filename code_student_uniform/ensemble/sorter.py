from sys import argv
from os import system
for filename in argv[1:]:
	cmd = 'sort -r %s -o %s'%(filename,filename)
	print cmd
	system(cmd)