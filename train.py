import cv2
import os
import numpy as np
from PIL import Image

def train_classifier():
    data_dir=("data")
    path=[os.path.join(data_dir,file)for file in os.listdir(data_dir)]
    faces=[]
    ids=[]
    for image in path:
        img=Image.open(image).convert('L')
        imageNp=np.array(img,'uint8')
        id=int(os.path.split(image)[1].split('.')[1])
        faces.append(imageNp)
        ids.append(id)
        cv2.imshow('training',imageNp)
        cv2.waitKey(1)==13
    ids=np.array(ids)
    ##Train and Save

    clf=cv2.face.LBPHFaceRecognizer.create()
    clf.train(faces,ids)
    clf.write("classifier.xml")
    cv2.destroyAllWindows()
    print("done")

train_classifier()