
import os
from os import listdir
from os.path import isfile, join


def get_current_time_as_string():
	#>>> from time import gmtime, strftime
	#strftime("%a, %d %b %Y %H:%M:%S", gmtime())
	#'Thu, 28 Jun 2001 14:17:15 +0000'
	from time import gmtime, strftime
	return strftime("%Y_%m_%d__%H_%M_%S", gmtime())

def get_last_file_with_ending(extension):
	mypath = "."
	
	files = [f for f in listdir(mypath) if isfile(join(mypath, f))]
	files = filter(lambda f: f.endswith(extension), files)

	try:
		latest_file = max(files, key=os.path.getctime)
		return latest_file
	except ValueError:
		return None
	

def get_last_weight_file():
	return get_last_file_with_ending(".h5")


#def get_last_weight_file():
#	mypath = "."
	
#	files = [f for f in listdir(mypath) if isfile(join(mypath, f))]
#	files = filter(lambda f: f.endswith(".h5"), files)

#	try:
#		latest_file = max(files, key=os.path.getctime)
#		return latest_file
#	except ValueError:
#		return None
	

	




