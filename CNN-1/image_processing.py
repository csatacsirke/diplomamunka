import numpy as np
import cv2

from log import log


import dataset_handler



default_input_file_name = 'jura/11.10/Bpas-Verdict.csv'


def normalize_image(image, corners, out_width = 1000, out_height = 1000):
	
	#image = np.array(pil_image)
	
	src = np.array(corners, dtype=np.float32);
	dst = np.array([[0,out_height], [out_width,out_height], [out_width,0], [0,0]], dtype=np.float32)


	transform = cv2.getPerspectiveTransform(src, dst)     
	image = cv2.warpPerspective(image, transform,(out_width, out_height))



	
	return image


"""
	dx = np.random.randint(-image.width//4, image.width//4+1)
	dy = np.random.randint(-image.height//4, image.height//4+1)

	
	center_x = image.width // 2 + dx
	center_y = image.height // 2 + dy
	# TODO randomizálni kicsit a centert

	left = center_x - img_width // 2
	top = center_y - img_height // 2
	right = left + img_width
	bottom = top + img_height
	
	image = image.crop((left, top, right, bottom))

	assert(image.width == img_width and image.height == img_height)
	"""



def crop_random(image, dimensions):
	height, width, channels = image.shape
	new_width, new_height = dimensions
	
	if width < new_width or height < new_height:
		log("Warning: image_processing.crop_random(image, dimensions): width < new_width or height < new_height")
		return image

	rnd = np.random.uniform()
	offset_x = int(rnd*(width - new_width))
	rnd = np.random.uniform()
	offset_y = int(rnd*(height - new_height))


	"""
	stackoverflow:
	If we consider (0,0) as top left corner of image called im with left-to-right as x direction and top-to-bottom as y direction.
	and we have (x1,y1) as the top-left vertex and (x2,y2) as the bottom-right vertex of a rectangle region within that image, then:

	roi = im[y1:y2, x1:x2]
	"""

	image = image[offset_y:offset_y+new_height, offset_x:offset_x+new_width]

	return image

def crop_top_right(image, dimensions):
	height, width, channels = image.shape
	new_width, new_height = dimensions

	if width < new_width or height < new_height:
		log("Warning: image_processing.crop_top_right(image, dimensions): width < new_width or height < new_height")
		return image
	
	offset_x = width - new_width
	offset_y = 0
	
	image = image[offset_y:offset_y+new_height, offset_x:offset_x+new_width]
	
	return image

def crop_center(image, ratio):
	height, width, channels = image.shape

	new_width = round(width * ratio)
	new_height = round(height * ratio)

	offset_x = (width - new_width) // 2
	offset_y = (height - new_height) // 2

	image = image[offset_y:offset_y+new_height, offset_x:offset_x+new_width]
	
	return image
	




def test():
	file_name = "d:\\diplomamunka\\spaceticket\\copy\\spactick_2017_09_11_13_24_47_503__0.png"
	image = cv2.imread(file_name)

	corners_list = dataset_handler.read_corners(default_input_file_name)
	corners = dataset_handler.find_corners_in_list(corners_list, file_name)

	if corners is None:
		assert("vaze" == "lofasz")
		return

	#	209.000000;1660.000000
	# 	1434.000000;1687.000000;
	# 	1461.000000;477.000000;
	# 	237.000000;441.000000;,


	src = np.array(corners, dtype=np.float32);
	dst = np.array([[0,500], [500,500], [500,0], [0,0]], dtype=np.float32)

	transform = cv2.getPerspectiveTransform(src, dst)     
	
	image = cv2.warpPerspective(image, transform,(500, 500))

	cv2.imwrite("kicsinyitett_proba.png", image)
	
	cv2.imshow('warp test', image)

	cv2.waitKey(0)
	cv2.destroyAllWindows()


def test2():
	file_name = "d:\\diplomamunka\\spaceticket\\copy\\spactick_2017_09_11_13_24_47_503__0.png"
	image = cv2.imread(file_name)
	
	image = crop_top_right(image, (800, 800))


	cv2.imshow('crop test', image)


	cv2.waitKey(0)
	cv2.destroyAllWindows()

	
def test3():
	file_name = "d:\\diplomamunka\\spaceticket\\copy\\normalized_spactick_2017_09_11_13_24_47_503__0.png"
	image = cv2.imread(file_name)
	
	image = crop_center(image, 0.6)


	cv2.imshow('crop center test', image)


	cv2.waitKey(0)
	cv2.destroyAllWindows()


if __name__ == "__main__":
	test3()




