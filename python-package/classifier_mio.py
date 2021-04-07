# Part of the code is extracted from https://github.com/davidsandberg/facenet
# Other from http://insightface.ai/build/examples_face_analysis/demo_analysis.html

import numpy as np
import argparse
import os
import sys
import math
import pickle
from sklearn.svm import SVC

import insightface
import cv2
import numpy as np

class ImageClass():
    "Stores the paths to images for a given class"
    def __init__(self, name, image_paths):
        self.name = name
        self.image_paths = image_paths
  
    def __str__(self):
        return self.name + ', ' + str(len(self.image_paths)) + ' images'
  
    def __len__(self):
        return len(self.image_paths)

def get_image_paths(facedir):
    image_paths = []
    if os.path.isdir(facedir):
        images = os.listdir(facedir)
        image_paths = [os.path.join(facedir,img) for img in images]
    return image_paths

def get_dataset(path, has_class_directories=True):
    dataset = []
    path_exp = os.path.expanduser(path)
    classes = [path for path in os.listdir(path_exp) \
                    if os.path.isdir(os.path.join(path_exp, path))]
    classes.sort()
    nrof_classes = len(classes)

    # print (nrof_classes)
    for i in range(nrof_classes):
        class_name = classes[i]
        facedir = os.path.join(path_exp, class_name)
        image_paths = get_image_paths(facedir)
        dataset.append(ImageClass(class_name, image_paths))
  
    return dataset

def get_embeddings(paths):

	embedding_size = 512

	nrof_images = len(paths)
	# print(nrof_images)
	
	embeddings = np.zeros((nrof_images, embedding_size))
	ctx_id = -1
	model = insightface.app.FaceAnalysis()
	model.prepare(ctx_id = ctx_id, nms=0.4)

	for i in range(nrof_images):
		# print(i,"/",nrof_images)
		img = cv2.imread(paths[i], cv2.IMREAD_COLOR)
		faces = model.get(img)

		# if (len(faces)>1):
			# Para comprobar que la cara principal siempre tiene idx0 (la más centrada)			
			# for idx, face in enumerate(faces):
				# print(idx, " ", face.bbox)
				# (x, y, z, zz) = face.bbox
				# cv2.putText(img, str(idx), (int(x), int(y)), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
				# cv2.rectangle(img, (int(x), int(y)), (int(z), int(zz)), (0, 255, 0), 2)
			# cv2.imshow("Wenas",img)
			# cv2.waitKey(100)

		embeddings[i] = faces[0].embedding
	return embeddings



def main(args):

	dataset = get_dataset(args.data_dir)

	paths = []
	labels = []
	for i in range(len(dataset)):
		paths += dataset[i].image_paths
		labels += [i] * len(dataset[i].image_paths)
	# print(image_paths)
	# print(labels)

	emb_array = get_embeddings(paths)
	# print(emb_array)
	classifier_filename_exp = os.path.expanduser(args.classifier_filename)


	if (args.mode=='TRAIN'):
		# Train classifier
		model = SVC(kernel='linear', probability=True)
		model.fit(emb_array, labels)

		# Create a list of class names
		class_names = [ cls.name.replace('_', ' ') for cls in dataset]

		# Saving classifier model
		with open(classifier_filename_exp, 'wb') as outfile:
			pickle.dump((model, class_names), outfile)
		print('Saved classifier model to file "%s"' % classifier_filename_exp)
                
	elif (args.mode=='CLASSIFY'):
		# Classify images
		with open(classifier_filename_exp, 'rb') as infile:
		    (model, class_names) = pickle.load(infile)


		predictions = model.predict_proba(emb_array)
		best_class_indices = np.argmax(predictions, axis=1)
		best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]

		indices = np.ones(len(best_class_indices), dtype=int)
		for i in range(len(best_class_indices)):
			# print(i)
			indice = int(paths[i].split("/")[-1].split(".jpg")[0])
			indices[indice] = int(i)

		for i in range(len(best_class_indices)):
                    print("\nFoto:\t"+ str(paths[indices[i]].split("/")[-1]))
                    pred = np.argsort(predictions[indices[i]])[::-1]
                    for j in range(5):#len(pred)):
                        print('%4d\t%s\t%.5f' % (j+1,class_names[pred[j]], predictions[indices[i]][pred[j]]))

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('mode', type=str, choices=['TRAIN', 'CLASSIFY'],
        help='Indicates if a new classifier should be trained or a classification ' + 
        'model should be used for classification', default='CLASSIFY')
    parser.add_argument('data_dir', type=str,
        help='Path to the data directory containing aligned LFW face patches.')
    # parser.add_argument('model', type=str, 
    #     help='Could be either a directory containing the meta_file and ckpt_file or a model protobuf (.pb) file')
    parser.add_argument('classifier_filename', 
        help='Classifier model file name as a pickle (.pkl) file. ' + 
        'For training this is the output and for classification this is an input.')
    # parser.add_argument('--use_split_dataset', 
        # help='Indicates that the dataset specified by data_dir should be split into a training and test set. ' +  
        # 'Otherwise a separate test set can be specified using the test_data_dir option.', action='store_true')
    # parser.add_argument('--test_data_dir', type=str,
        # help='Path to the test data directory containing aligned images used for testing.')
    parser.add_argument('--batch_size', type=int,
        help='Number of images to process in a batch.', default=90)
    parser.add_argument('--image_size', type=int,
        help='Image size (height, width) in pixels.', default=160)
    parser.add_argument('--seed', type=int,
        help='Random seed.', default=666)
    parser.add_argument('--min_nrof_images_per_class', type=int,
        help='Only include classes with at least this number of images in the dataset', default=20)
    parser.add_argument('--nrof_train_images_per_class', type=int,
        help='Use this number of images from each class for training and the rest for testing', default=10)
    
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
