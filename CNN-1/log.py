from util import get_current_time_as_string

import os

log_file_name = "log_" + get_current_time_as_string() + ".txt"

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def set_log_file_path(new_file_path):
	global log_file_name
	log_file_name = new_file_path
	pass



def log(*args, level=None):
	for arg in args: 
		log_message(str(arg), ending=" ", level=level)
	log_message("") # sortores miatt

def log_message(message, ending="\r\n", level=None):
	
	if level is "warning":
		#os.system('color 4')
		pass

	print(message, end=ending)

	if level is not None:
		#os.system('color 0')
		pass

	with open(log_file_name, 'a') as log_file:
		log_file.write(message + ending)



