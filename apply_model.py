import tkinter as tk
import cv2
import pickle
import numpy as np
from sklearn.svm import SVC
from PIL import Image
from PIL import ImageTk
from tkinter import ttk
from tkinter import messagebox
from tkinter import filedialog as fd
from tkinter.filedialog import askopenfile

root = tk.Tk()
root.title('Leaf Classification')
root.resizable(True, True)
root.geometry('1366x768')

frame = tk.Frame(root, width = 1300, height = 700)
frame.pack(pady=(10,0))

labelA = tk.LabelFrame(frame, text = "Original Image", width=650, height=600)
labelA.pack(side='left')
panelA = None
labelB = tk.LabelFrame(frame, text = "Canny Edge Detection", width=650, height=600)
labelB.pack(side='right')
panelB = None

def select_image():

    global panelA, panelB, path

    filetypes = (
        ('Image Files', '*.jpg'),
        ('Image Files', '*.jpeg'),
        ('Image Files', '*.png'),
        ('All Files', '*.*')
    )

    path = fd.askopenfilename(
        title = 'Select Image',
        filetypes=filetypes
    )

    if len(path) > 0:
        image = cv2.imread(path)
        
        image = cv2.resize(image, (650,500), interpolation = cv2.INTER_AREA)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edged = cv2.Canny(gray, 400, 400)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        image = Image.fromarray(image)
        edged = Image.fromarray(edged)

        image = ImageTk.PhotoImage(image)
        edged = ImageTk.PhotoImage(edged)

        if panelA is None or panelB is None:
            panelA = tk.Label(labelA,image=image, width=650, height=577)
            panelA.image = image
            panelA.pack()

            panelB = tk.Label(labelB,image=edged, width=650, height=577)
            panelB.image = edged
            panelB.pack()
        else:
            panelA.configure(image=image)
            panelB.configure(image=edged)
            panelA.image = image
            panelB.image = edged

def predict():
    categories1 = ['Daun Ara Suci','Daun Bayam Hijau','Daun Bayam Malabar','Daun Buah Samarinda','Daun Cendana','Daun Delima', 'Daun Ficus Auriculata','Daun Jamblang','Daun Jambu Biji','Daun Jambu Mawar', 'Daun Jeruk Sitrun', 'Daun Jintan', 'Daun Kelabat', 'Daun Kelor', 'Daun Kembang Sepatu', 'Daun Kersen', 'Daun Lengkuas', 'Daun Malapari', 'Daun Mangga', 'Daun Melati', 'Daun Mimba', 'Daun Mint', 'Daun Mondokaki', 'Daun Nangka', 'Daun Oleander', 'Daun Ruku-Ruku', 'Daun Salam Koja', 'Daun Sesawi India', 'Daun Sirih', 'Daun Srigading']

    data = open('model.sav','rb')
    model = pickle.load(data)
    data.close()

    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    resize = cv2.resize(img,(200,200))
    edge = cv2.Canny(resize, 400, 400)
    pred = [edge.flatten()]

    messagebox.showinfo("Hasil Prediksi",f"Diprediksi sebagai daun: {categories1[model.predict(pred)[0]]}")

buttonOpen = ttk.Button(
    root,
    text='Select an image',
    command=select_image
)

buttonPred = ttk.Button(
    root,
    text='Predict',
    command=predict
)

buttonOpen.place(
    x=300,
    y=600
)

buttonPred.place(
    x=1000,
    y=600
)
root.mainloop()