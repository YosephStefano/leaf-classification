import pickle
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

pick_in = open('canny.pickle','rb')
data = pickle.load(pick_in)
pick_in.close()

features = []
labels = []

for feature, label in data:
    features.append(feature)
    labels.append(label)

xtrain, xtest, ytrain, ytest = train_test_split(features, labels, test_size=0.20)

model = SVC(C=10, kernel='rbf')
model.fit(xtrain,ytrain)

pick = open('model.sav','wb')
pickle.dump(model,pick)
pick.close()