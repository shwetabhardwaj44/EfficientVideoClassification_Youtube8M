import fnmatch
import os
import sys
from subprocess import call

root_directory = '../models/'
directory = './'
index = 0

# frame_level_Summary_mode
# Checking the validity of the arguments

if(len(sys.argv) != 3):
	print "Wrong number of arguments entered.\n\nThe correct format is: python delete.py directory_name index"
	sys.exit()
else:
	directory = root_directory + sys.argv[1] + "/"
	index = sys.argv[2]

	if (not os.path.exists(directory)):
		print "The given directory does not exist."
		sys.exit()

	if (not index.isdigit()):
		print "The given index is not a positive integer.\nFormat: Enter the integer without a sign"
		sys.exit()

	index = int(index)

	if index < 0:
		print 'Terminating program as negative index given.'
		sys.exit()

# Collecting all files in the directory (which start with model)
files = []
for file in os.listdir(directory):
    if fnmatch.fnmatch(file, 'model*'):
    	files.append((file, int((file.split('.')[1]).split('-')[1])))

# print files

# Collecting all files with index less than of equal to the given index
files_deleting = [f for f in files if f[1] <= index]


if len(files_deleting) == 0:
	print 'no file to delete'
	sys.exit()

print "\nThe following files will be deleted:\n"
for f in files_deleting:
	print f[0]


print "\nEnter 1 to delete all listed files.\n"

choice = raw_input("Enter a number: ")

if (not choice.isdigit()) or int(choice) != 1:
	print 'Looks like you do not want to delete any of the files!\nNo files were deleted.'
else:
	for f in files_deleting:
		# print directory + f[0], os.path.exists(directory + f[0])
		os.remove(directory + f[0])
