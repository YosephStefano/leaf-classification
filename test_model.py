import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import train_test_split

categories = ['Daun Ara Suci','Daun Bayam Hijau','Daun Bayam Malabar','Daun Buah Samarinda','Daun Cendana','Daun Delima', 'Daun Ficus Auriculata','Daun Jamblang','Daun Jambu Biji','Daun Jambu Mawar', 'Daun Jeruk Sitrun', 'Daun Jintan', 'Daun Kelabat', 'Daun Kelor', 'Daun Kembang Sepatu', 'Daun Kersen', 'Daun Lengkuas', 'Daun Malapari', 'Daun Mangga', 'Daun Melati', 'Daun Mimba', 'Daun Mint', 'Daun Mondokaki', 'Daun Nangka', 'Daun Oleander', 'Daun Ruku-Ruku', 'Daun Salam Koja', 'Daun Sesawi India', 'Daun Sirih', 'Daun Srigading']

pick_in = open('canny.pickle','rb')
data = pickle.load(pick_in)
pick_in.close()

features = []
labels = []

for feature, label in data:
    features.append(feature)
    labels.append(label)

xtrain, xtest, ytrain, ytest = train_test_split(features, labels, test_size=0.20)

pick = open('model.sav','rb')
model = pickle.load(pick)
pick.close()

prediction = model.predict(xtest)
accuracy = model.score(xtest, ytest)

print(f'Akurasi: {accuracy*100}%')
print('Prediksi: ',categories[prediction[0]])

leaf = xtest[0].reshape(200,200)
plt.imshow(leaf,cmap='gray')
plt.show()