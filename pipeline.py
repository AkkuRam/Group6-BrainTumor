import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

class ImagePipeline:

    def __init__(self, path, size):
        self.path = path
        self.size = size
    
    def load_images(self):
        classes = ['glioma/', 'meningioma/', 'notumor/', 'pituitary/']
        images = []
        labels = []
        for index, c in enumerate(classes):
            class_path = os.path.join(self.path, c)
            for name in os.listdir(class_path):
                img_path = os.path.join(class_path, name)
                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img = cv2.resize(img, (self.size, self.size))
                images.append(img)
                labels.append(index)
        return np.array(images), np.array(labels)
    
    def preprocess(self, images):
        for i in range(len(images)):
            images[i] = cv2.normalize(images[i], None, 0, 255, cv2.NORM_MINMAX)
            images[i] = cv2.GaussianBlur(images[i], (7,7), 0)
        return images
    
    def get_final_images(self):
        train_images, train_labels = self.load_images()
        train_images = pipe.preprocess(train_images)
        return train_images, train_labels


path = "dataset/Training/"
pipe = ImagePipeline(path, 256)
train_images, train_labels = pipe.get_final_images()
cv2.imwrite('preprocessed.jpg', train_images[0])