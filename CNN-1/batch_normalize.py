import os
import argparse
import cv2


import image_processing
import dataset_handler

overwrite = False


def test():

	mypath = "d:\\Diplomamunka\\CNN-1\\working_dir\\"
	f = []
	for (dirpath, dirnames, filenames) in os.walk(mypath):
		
		print(dirpath, dirnames)
	


def main():

	if False:
		parser = argparse.ArgumentParser(description='Recursive normalizer.')
		parser.add_argument('csv_in', metavar='csv_in', type=str)
		#parser.add_argument('dir_out', metavar='dir_out', type=str)

		args = parser.parse_args()

		csv_in = args.csv_in
		#dir_out = args.dir_out
	else:
		csv_in = "d:\\Diplomamunka\\CNN-1\\working_dir\\jura\\11.10\\Bpas-Verdict.csv"
	#print(dir_in, dir_out)

	#if not os.path.isfile(csv_in) or not os.path.isdir(dir_out):
	if not os.path.isfile(csv_in):
		print(csv_in, " does not exist")
		raise AssertionError

	#if dir_in == dir_out:
	#	raise Exception

	#corners_list = []
	input_list = dataset_handler.read_full_input(csv_in, zip_results=True)


	for entry in input_list:
		path, _, corners = entry

		name = os.path.basename(path)
		dir = os.path.dirname(path)

		if len(corners) == 0:
			print(path, " skipped: no corner info")
			continue
		
		new_name = "normalized_" + name
		new_path = dir + "\\" + new_name

		in_image = cv2.imread(path)
		result = image_processing.normalize_image(in_image, corners)
		if not os.path.exists(new_path) or overwrite:
			cv2.imwrite(new_path, result)
			print(path, "->", new_name , " done")
		else:
			print(new_path, "already exists")

		pass
	


	return

if __name__ == "__main__":
	
	try:
		main()
	except AssertionError:
		print("Task failed")



