import csv
import numpy as np
import sys
import os
import gc

import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow


# saját dolgok
from log import log
from log import set_log_file_path
from util import get_current_time_as_string
from util import get_last_weight_file
import image_processing

import dataset_handler
import statistics 


########################################################################################
# TODO
# kb 1600x1600 as képeink vannak, kicsinyiteni ha nem muszáj nem akarunk
# valszeg majd azt kéne csinálni hogy egy fix méretet középröl kivágunk
"""
 1) kicsinyitás ( downscale ) !!!!
 2) 
	random kivágás
	fix kivágás
		-középről
		-sarkokról
		
col2im / im2col
256->200 : valami ilyen arányt ide is 
5 rész (sarkok+közép, 1 háló 5x példával) : autoencodernél háló predikciók összeátlagolása, osztályozónál meg mindegyikre ugyanaz a címke
            		
debug: 1-2 képre overfit a trainen

*TODO a transzformációt elöre kéne megcsinálni, nem röptében

TODO kéne egy olyan futtatási opció amivel csak predikálni lehet
"""


#base_dir = 'd:/diplomamunka/SpaceTicket_results/'
#default_input_file_name = base_dir + 'Bpas-Verdict.csv'


#default_input_file_name = 'jura/11.14/Bpas-Verdict.csv'
default_input_file_name = "jura\\2017.10.25\\Bpas-Verdict.csv"

g_img_height, g_img_width = 256, 256



# tanulási paraméterek
epochs = 50
g_batch_size = 16
# g_batch_size = 16

use_autoencoder = True
recalculate = True
stage = 1

#evaluate_only = False
weights_only = False
# Ezt töltsd ki ha szeretnéd betölteni a munkamenetet
weight_file = None



evaluate_show_pictures = True
warn_for_no_corner_info = False
use_normalized_images = True
use_full_dataset_for_test = False

#if not recalculate:
#	weight_file = get_last_weight_file()

	

#if evaluate_only:
#	assert(recalculate == False)
#	recalculate = False



if stage == 2:
	assert(recalculate == False)
	recalculate = False

	
#fnCopy = "d:\\diplomamunka\\Temp\\kicsi\\SPACTICK_2017_09_18_15_43_35_942_Marci.eredeti_1___.png"
#fnOrig = "d:\\diplomamunka\\Temp\\kicsi\\SPACTICK_2017_10_09_16_02_27_938_Marci.fenymasolt.copy_0___.png"

#weight_file = "weights/2017_10_17__12_28_56.h5"



########################################################################################


#result_dir = "weights/"
# ha van rendes working directory akkor erre nincs szükség
#result_dir = "" 
#log_file_path = result_dir + get_current_time_as_string() + "_log" + ".txt"
#set_log_file_path(log_file_path)

log("Loading keras - this may be slow...", level="warning")


import tensorflow as tf

# TODO megcsinálni
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose, UpSampling2D, ZeroPadding2D, Cropping2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from keras.callbacks import History, Callback
from keras.models import load_model

# todo kidobni
import keras

import keras.preprocessing.image as KerasImage
from PIL import Image as PilImage
import cv2

# a rendszer nem engedi lefoglalni az egész memoriát, viszont a 
# tensorflow megprobalja
# kb 0.3 ra mindig jó volt
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.7
session = tf.Session(config=config)
set_session(session)


# ez amúgy jó eséllyel nem kell mert alapból channels last, 
# és ehhez alkalmazkodtunk
# TODO ezt jó eséllyel ki is lehet hagyni az összes image-könyvtárral együtt,
# mivel már opencv-t hasznuálunk betöltésre is
if K.image_data_format() == 'channels_first':
    input_shape = (3, g_img_height, g_img_width)
else:
    input_shape = (g_img_height, g_img_width, 3)

log("Done loading keras")


log("Start")

# globalis változok
# todo majd a végén ezeket pl command line paraméternek


def process_image(image, target_dims=None, training_phase=True):
	image = image_processing.crop_center(image, 0.6)
	if use_autoencoder:
		if training_phase:
			image = image_processing.crop_random(image, target_dims)
	else:
		image = image_processing.crop_top_right(image, target_dims)
	
		
	return image
	


def load_image_as_array(file_name):
	#image = KerasImage.load_img(file_name)
	image = cv2.imread(file_name).astype(float)/255.0
	# corners_list -> global
	
	#image_array = KerasImage.img_to_array(image)
	#return image_array
	return image



#use_normalized_images
def create_single_sample_generator(file_names, ground_truths, target_dims, training_phase=True):
	assert(len(ground_truths) == len(ground_truths))
	count = len(file_names)
	while True:
		permutation = np.random.permutation(count)
		for i in permutation:
			file_name = file_names[i]
			#normalized_file_name = dataset_handler.get_normalized_counterpart(file_name)
			if not os.path.exists(file_name):
				continue

			#image_array = load_image_as_array(normalized_file_name)
			image_array = load_image_as_array(file_name)

			image_array = process_image(image_array, target_dims=target_dims, training_phase=training_phase)

			if image_array is None:
				continue
			
			yield image_array, ground_truths[i]
	


# globalis parameter a g_batch_size
def create_generator(file_names, ground_truths, target_dims=(g_img_width, g_img_height), use_autoencoder=False, training_phase=True, batch_size=g_batch_size):

	assert(len(ground_truths) == len(ground_truths))
	single_sample_generator =  create_single_sample_generator(file_names, ground_truths, target_dims, training_phase=training_phase)




	while True:
		
		x = None
		images = []
		y = np.empty([batch_size, 1])

		for i in range(batch_size):
			
			if use_autoencoder:
				ground_truth = 0
				while ground_truth == 0:
					image_array, ground_truth = next(single_sample_generator)
			else:
				image_array, ground_truth = next(single_sample_generator)

			if x is None:
				height, width, channels = image_array.shape
				x = np.empty([batch_size, height, width, channels])

			x[i] = image_array
			y[i] = ground_truth


			#images.append(image_array)
			#np.append(x, image_array, axis=0)
			#np.append(y, ground_truth, axis=0)
		#x = np.array(images)
		if use_autoencoder:
			yield x,x
		else:
			yield x,y




def beautfy_result(results_row):
	"""
	Alapból a str(np.array()) valami ilyet ir ki:
	[[0]] egy sima 0 helyett
	"""
	#y_pred = round(y_pred.flatten().flatten().tolist()[0])
	file_name, y, y_pred = results_row
	y_pred = round(y_pred.flatten().flatten().tolist()[0])
	return (file_name, y, y_pred)



def evaluate():
	assert(weight_file is not None)

	log("Evaluating...")
	#validation_file_names, validation_ground_truths,
	#test_generator = create_generator(test_file_names, test_ground_truths)
	eval_full_dataset = True
	if eval_full_dataset:
		file_names, ground_truths = dataset_handler.read_full_input(default_input_file_name);
	else:
		file_names = test_file_names
		ground_truths = test_ground_truths

	#length = len(test_file_names)
	length = len(file_names)

	results = []

	for i in range(length):
		file_name = file_names[i]
		#file_name = test_file_names[i]
		
		image = load_image_as_array(file_name)
		height, width, channels = image.shape
		x = image.reshape((1, height, width, channels))
		if x is None:
			#log("Skipping file (no corners):", file_name)
			continue
		#y = test_ground_truths[i]
		y = ground_truths[i]
		y_pred = model.predict(x)

		results.append((file_name, y, y_pred))


	# Alapból a str(np.array()) valami ilyet ir ki:
	# [[0]] egy sima 0 helyett
	results = map(beautify_results, results)
	header = ("file", "igazság", "jóslat")
	evaluate.write_results_to_csv(results, header=header)


# https://stackoverflow.com/questions/12201577/how-can-i-convert-an-rgb-image-into-grayscale-in-python
def rgb2gray(rgb):
	
	r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
	#gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
	gray = 0.3333 * r + 0.3333 * g + 0.3333 * b
	
	return gray

def show_pictures(x, y, prediction):
	
	log(np.shape(prediction))

	fig = plt.figure()

	#img = img[:,:,::-1]
	#red, green, blue, alpha = data.T 
	#data = np.array([blue, green, red, alpha])

	subplot = fig.add_subplot(1,3,1)
	x = x / 255.0
	img = np.subtract(1.0, x)
	
	plt.imshow(img)
	#red, green, blue = x.T 
	#img = np.array([blue, green, red])
	#img = img.transpose()
	#plt.imshow(img)

	subplot = fig.add_subplot(1,3,2)
	prediction = np.divide(prediction, 255)
	plt.imshow(prediction)

	subplot = fig.add_subplot(1,3,3)
	diff = np.abs(prediction - x)
	plt.imshow(diff)

	#subplot = fig.add_subplot(1,3,3)
	#gray = rgb2gray(prediction)
	#plt.imshow(gray)
	fig.savefig("images/" + get_current_time_as_string() + "_teszt__.png", dpi=600)
	#plt.show()
	plt.close(fig)

def np_wrap(image):
	h, w, ch = image.shape
	return image.reshape((1, h, w, ch))

def np_unwrap(image):
	dims, h, w, ch = image.shape
	assert(dims == 1)
	return image.reshape((h, w, ch))

def show_picutures_looped():
	while True:
		samples = next(training_generator)
		for sample in samples:
			x, y = sample
			prediction = model.predict(x)

			show_pictures(x, y, prediction)
	

def evaluate_autoencoder():
	
	#gen = create_generator([fnOrig], [1])
	if evaluate_show_pictures:
		show_picutures_looped()
	
	

	return






class Scheme:
	def __init__(self, postfix="generic"):
		self.postfix = postfix
	

	def load_or_build_model(self, force_recalculate=False):
		model = None
		#if force_recalculate:
		weight_file = None if force_recalculate else get_last_weight_file(postfix=self.postfix)

		log("Weight file :", weight_file)

		if weight_file is not None:
			try:
				model = load_model(weight_file)
				log("Successfully loaded full model: ", weight_file, "\r\n")
			except ValueError:
				log("Invalid model file (probably only saved weights)", weight_file, "\r\n")
				model = None

		if model is None:
			model = self.build_model()
			log("Rebuilt model from scratch")
			#if use_autoencoder:
			#	model = build_autoencoder_model()
			#else:
			#	model = build_predictor_model()


			if weight_file is not None:
				log("Loading weights from: " + weight_file)
				try:
					model.load_weights(weight_file)
				except:
					log("Failed to load weights!!!! probably different architecture")
		return model


	def save_model(self, model):
		postfix = self.postfix;
		postfix = "" if postfix=="" else "." + postfix

		if weights_only:
			file_name = get_current_time_as_string() + '.weights' + postfix +'.h5'
			model.save_weights(file_name)
		else:
			file_name = get_current_time_as_string() + '.full_model' + postfix +'.h5'
			model.save(file_name)

		log()
		log("Saved weights to : " + file_name)



class AutoencoderScheme(Scheme):
	
	def __init__(self):
		super().__init__(postfix="autoencoder")


	"""
	float16 ot meg lehet próbálni
	conv2 + dense:
		kernel_regularizer
		keras.regularizers.l2(0.001) % a hiperparamétert is be kell még  löni
	
		autoencoder:
			conv2dtranspose
			loss: squared error
			tanitás csak az eredetiken
			érdemes kézzel megnézni hogy a predict milyen képet generál
	
	"""

	def build_model(self):
	
		model = Sequential()
		"""
		model.add(Conv2D(64, (11, 11), input_shape=input_shape, padding="same", data_format="channels_last"))
		model.add(Conv2DTranspose(3, (11, 11), padding="same", data_format="channels_last"))
		"""
	
		log("input shape: ", input_shape)

		#kernel_sizes = [13, 13, 13]
		kernel_sizes = [5, 5, 5]
		kernel_sizes = [7, 7, 7]
		kernel_dims = list(map(lambda x: (x,x), kernel_sizes))
		iterator = iter(kernel_dims)
		reverse_iterator = reversed(kernel_dims)

		padding = "same"
		#padding = "valid"

		#model.add(Conv2D(64, (5, 5), input_shape=input_shape))

		padding_size = sum(kernel_sizes)*2
	
		model.add(ZeroPadding2D(padding=(padding_size, padding_size), input_shape=(None, None, 3)))
		model.add(Conv2D(64, next(iterator), padding=padding))
		model.add(Activation('relu'))
		model.add(MaxPooling2D(pool_size=(2, 2)))

		model.add(Conv2D(128, next(iterator), padding=padding))
		model.add(Activation('relu'))
		model.add(MaxPooling2D(pool_size=(2, 2)))


		model.add(Conv2D(256, next(iterator), padding=padding))
		model.add(Activation('relu'))

		#model.add(MaxPooling2D(pool_size=(2, 2)))
	
		# dense layer lehet dense is
	
		#####
	

		model.add(Conv2DTranspose(128, next(reverse_iterator), padding=padding))
		model.add(Activation('relu'))
		model.add(UpSampling2D(size=(2, 2)))
	

		model.add(Conv2DTranspose(64, next(reverse_iterator), padding=padding))
		model.add(Activation('relu'))
		model.add(UpSampling2D(size=(2, 2)))
	
		model.add(Conv2DTranspose(3, next(reverse_iterator), padding=padding))

		model.add(Cropping2D(cropping=(padding_size, padding_size)))

		#model.add(ZeroPadding2D())
		#model.add(Activation('relu'))
		#model.add(UpSampling2D(size=(2, 2), data_format="channels_last"))
	
		#####

		model.compile(loss='mean_squared_error',
					  optimizer='adam', # ADAM
					  metrics=['accuracy']) 


		shape = model.get_output_shape_at(0)
		log(shape)
		#log(K.int_shape(last_index - 1))
		log("")
		log("")
		log("")
		return model
	
	def eval(self, model):
		
		entries = list(zip(training_file_names, training_ground_truths))
		np.random.shuffle(entries) # in-place

		#entries = entries[0:50]

		results = []

		batch_size = 20
		for i in range(0, len(entries), batch_size):


			batch = entries[i:i+batch_size]
			x = []
			for entry in batch:
				file_name, y = entry
				image = load_image_as_array(file_name)
				image = process_image(image, training_phase=False)
				x.append(image)


			x = np.array(x)
			predictions = model.predict(x)

			for index, prediction in enumerate(predictions):
			
				x_linear = x[index].reshape(-1)
				prediction_linear = prediction.reshape(-1)
				#MSE_np = keras.losses.mean_squared_error(x_linear, prediction_linear)
				MSE_np = statistics.mean_squared_error(x_linear, prediction_linear)

				MSE = MSE_np
				#MSE = MSE_np.eval(session=session)
				#
				file_name, y = batch[index]
				##record = (x, prediction, y, file_name)
				record = (MSE, y, file_name)
				results.append(record)

		
			log(i, " / ", len(entries))

			gc.collect()

	
		#statistics.eval(results)
		statistics.write_results_to_csv(results, postfix="."+self.postfix+".stage2")


class PredictorScheme(Scheme):
	def __init__(self):
		super().__init__(postfix="predictor")

	def build_model(self):
		
		model = Sequential()
		model.add(Conv2D(64, (5, 5), input_shape=input_shape))
		model.add(Activation('relu'))
		model.add(MaxPooling2D(pool_size=(2, 2)))

		model.add(Conv2D(64, (5, 5)))
		model.add(Activation('relu'))
		model.add(MaxPooling2D(pool_size=(2, 2)))


		model.add(Conv2D(256, (5, 5)))
		model.add(Activation('relu'))
		model.add(MaxPooling2D(pool_size=(2, 2)))

		model.add(Conv2D(256, (5, 5)))
		model.add(Activation('relu'))
		model.add(MaxPooling2D(pool_size=(2, 2)))




		# 64 kicsi + lehet több dense egymás után
		model.add(Flatten())
		model.add(Dense(256))
		model.add(Activation('relu'))

		model.add(Dense(256))
		model.add(Activation('relu'))

		model.add(Dense(256))
		model.add(Activation('relu'))


		# lehet dropout a convok előtt is ( mindegyik előtt)
		model.add(Dropout(0.5))
		model.add(Dense(2))#1))
		model.add(Activation('softmax'))#'sigmoid'))

		model.compile(loss='sparse_categorical_crossentropy', #'binary_crossentropy',
					  optimizer='adam', #'rmsprop', # ADAM
					  metrics=['accuracy']) 

		return model

	def eval(self, model):
		
		if use_full_dataset_for_test:
			file_namas = training_file_names + validation_file_names + test_file_names
			ground_truths = training_ground_truths + validation_ground_truths + test_ground_truths
	
			entries = list(zip(file_namas, ground_truths))
		else:
			entries = list(zip(test_file_names, test_ground_truths))

		#entries = list(zip(training_file_names, training_ground_truths))
		#np.random.shuffle(entries) # in-place

		#entries = entries[0:50]

		results = []

		batch_size = 20
		for i in range(0, len(entries), batch_size):


			batch = entries[i:i+batch_size]
			x = []
			for entry in batch:
				file_name, y = entry
				image = load_image_as_array(file_name)
				image = process_image(image, training_phase=False, target_dims=(g_img_width, g_img_height))
				x.append(image)


			x = np.array(x)
			predictions = model.predict(x)

			for index, prediction in enumerate(predictions):
			
				file_name, y = batch[index]
				record = (prediction[1], y, file_name)
				results.append(record)

		
			log(i, " / ", len(entries))

			gc.collect()

	
		#statistics.eval(results)
		statistics.write_results_to_csv(results, postfix="."+self.postfix+".stage2")



#def load_or_build_model():
#	model = None
#	if weight_file is not None:
#		try:
#			model = load_model(weight_file)
#			log("Successfully loaded full model: ", weight_file, "\r\n")
#		except ValueError:
#			log("Invalid model file (probably only saved weights)", weight_file, "\r\n")
#			model = None

#	if model is None:
#		if use_autoencoder:
#			model = build_autoencoder_model()
#		else:
#			model = build_predictor_model()


#		if weight_file is not None:
#			log("Loading weights from: " + weight_file)
#			try:
#				model.load_weights(weight_file)
#			except:
#				log("Failed to load weights!!!! probably different architecture")
#	return model




# https://stackoverflow.com/questions/37293642/how-to-tell-keras-stop-training-based-on-loss-value
class EarlyStoppingByLossVal(Callback):
	def __init__(self, monitor='val_loss'):
		super(Callback, self).__init__()

		self.monitor = monitor
		self.lowest_loss = None
		self.consequitive_plateaus = 0
		self.max_consequitive_plateaus = 3
			
	def on_epoch_end(self, epoch, logs={}):
		current_loss = logs.get(self.monitor)
		if current_loss is None:
			warnings.warn("Early stopping requires %s available!" % self.monitor, RuntimeWarning)
			return
		if self.lowest_loss is None:
			self.lowest_loss = current_loss


		if self.lowest_loss < current_loss:
			self.consequitive_plateaus = self.consequitive_plateaus + 1

			log()
			log("Warning: plateau %d" % self.consequitive_plateaus)
			

			if self.consequitive_plateaus >= self.max_consequitive_plateaus:
				log("Early stopping")
				self.model.stop_training = True
		else:
			self.consequitive_plateaus = 0
			self.lowest_loss = current_loss


class SaveRegularly(Callback):
	def __init__(self, monitor='val_loss'):
		super(Callback, self).__init__()

		self.save_interval = 2
		self.counter = 0
			
	def on_epoch_end(self, epoch, logs={}):
		self.counter = self.counter + 1

		if self.counter == self.save_interval:
			self.counter = 0
			scheme.save_model(self.model)
			#save_model(self.model)



def save_model(model):
	
	if weights_only:
		file_name = get_current_time_as_string() + '.weights.h5'
		model.save_weights(file_name)
	else:
		file_name = get_current_time_as_string() + '.full_model.h5'
		model.save(file_name)

	log()
	log("Saved weights to : " + file_name)





################################################################################################
	
(training_file_names, training_ground_truths,
	validation_file_names, validation_ground_truths,
	test_file_names, test_ground_truths) = dataset_handler.load_dataset_or_create_new(default_input_file_name)


training_generator = create_generator(training_file_names, training_ground_truths, use_autoencoder=use_autoencoder)
validation_generator = create_generator(validation_file_names, validation_ground_truths, use_autoencoder=use_autoencoder)


if use_autoencoder:
	scheme = AutoencoderScheme()
else:
	scheme = PredictorScheme()


model = scheme.load_or_build_model(force_recalculate=recalculate)
#model = load_or_build_model()


#if evaluate_only:
#	log("Evaluating")
#	if use_autoencoder:
#		evaluate_autoencoder()
#	else:
#		evaluate()
#	sys.exit()
	
if stage is 1:
	log("Fitting")
	while True:
		"""
		callbacks:
			*early stopping 
			terminate on nan
			history
			*modell checkpoint
		itt is használni a class weightset
		"""
		# todo


		callbacks = [
			EarlyStoppingByLossVal(), SaveRegularly()
		]

		
		nb_train_samples = len(training_file_names)
		nb_validation_samples = len(validation_file_names)
		steps_per_epoch = nb_train_samples // g_batch_size
		validation_steps = nb_validation_samples // g_batch_size

		history = model.fit_generator(
			training_generator,
			steps_per_epoch=steps_per_epoch,
			epochs=epochs,
			validation_data=validation_generator,
			validation_steps=validation_steps,
			callbacks=callbacks)


		#save_model(model)

		#  ez legyen callbackban
		log(str(history.history['val_loss']))
		log(str(history.history['val_acc']))
		log("=============================================")
	
		#if use_autoencoder:
		#	evaluate_autoencoder()

		break

	pass
	log("Starting stage 2 ")
	stage = 2
	

if stage is 2:
	
	scheme.eval(model)
	"""
	entries = list(zip(training_file_names, training_ground_truths))
	np.random.shuffle(entries) # in-place

	#entries = entries[0:50]

	results = []

	batch_size = 20
	for i in range(0, len(entries), batch_size):


		batch = entries[i:i+batch_size]
		x = []
		for entry in batch:
			file_name, y = entry
			image = load_image_as_array(file_name)
			image = process_image(image, training_phase=False)
			x.append(image)


		x = np.array(x)
		predictions = model.predict(x)

		for index, prediction in enumerate(predictions):
			
			x_linear = x[index].reshape(-1)
			prediction_linear = prediction.reshape(-1)
			#MSE_np = keras.losses.mean_squared_error(x_linear, prediction_linear)
			MSE_np = statistics.mean_squared_error(x_linear, prediction_linear)

			MSE = MSE_np
			#MSE = MSE_np.eval(session=session)
			#
			file_name, y = batch[index]
			##record = (x, prediction, y, file_name)
			record = (MSE, y, file_name)
			results.append(record)

		
		log(i, " / ", len(entries))

		gc.collect()

	
	#statistics.eval(results)
	statistics.write_results_to_csv(results, postfix=".stage2")

	"""

##################################################################################################################


print("vege");



#log("Fitting[Test]")
#first, first_y = next(training_generator)
#first = np.array(first).reshape((1, g_img_height, g_img_width, 3))
#first_y = np.array(first_y).reshape((1,1))
#model.fit(first, first_y)
#log("Fitting[Test] - done")
#sys.exit()






