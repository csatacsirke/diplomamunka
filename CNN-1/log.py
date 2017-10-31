from util import get_current_time_as_string

log_file_name = "log_" + get_current_time_as_string() + ".txt"

def set_log_file_path(new_file_path):
	global log_file_name
	log_file_name = new_file_path
	pass



def log(*args):
	for arg in args: 
		log_message(str(arg), ending=" ")
	log_message("") # sortores miatt

def log_message(message, ending="\r\n"):
	print(message, end=ending)
	with open(log_file_name, 'a') as log_file:
		log_file.write(message + ending)



