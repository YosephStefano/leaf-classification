import os
import numpy as np
import cv2
import pickle
import random

dir = 'C:\\Users\\Joseph-PC\\Downloads\\Compressed\\Dataset\\leaf-edge\\leaf dataset\\Train'

categories = ['Daun Ara Suci','Daun Bayam Hijau','Daun Bayam Malabar','Daun Buah Samarinda','Daun Cendana','Daun Delima', 'Daun Ficus Auriculata','Daun Jamblang','Daun Jambu Biji','Daun Jambu Mawar', 'Daun Jeruk Sitrun', 'Daun Jintan', 'Daun Kelabat', 'Daun Kelor', 'Daun Kembang Sepatu', 'Daun Kersen', 'Daun Lengkuas', 'Daun Malapari', 'Daun Mangga', 'Daun Melati', 'Daun Mimba', 'Daun Mint', 'Daun Mondokaki', 'Daun Nangka', 'Daun Oleander', 'Daun Ruku-Ruku', 'Daun Salam Koja', 'Daun Sesawi India', 'Daun Sirih', 'Daun Srigading']

data = []

for category in categories:
    path = os.path.join(dir,category)
    label = categories.index(category)

    for img in os.listdir(path):
        imgpath = os.path.join(path,img)
        leaf_img=cv2.imread(imgpath, cv2.IMREAD_GRAYSCALE)
        try:
            resize = cv2.resize(leaf_img,(200,200))
            edged = cv2.Canny(resize, 100, 200)
            img = np.array(edged).flatten()
            data.append([img,label])
        except Exception as e:
            pass

random.shuffle(data)

pick_in = open('canny.pickle','wb')
pickle.dump(data,pick_in)
pick_in.close()