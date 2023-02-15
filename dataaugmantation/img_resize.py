import cv2, glob, os, shutil

trainPath = r""
classesPath = glob.glob(trainPath+"\*")
for classPath in classesPath:
    for imgPath in glob.glob(classPath+"\*"):
        img = cv2.imread(imgPath)
        img = cv2.resize(img, (224, 224))
        cv2.imwrite(imgPath, img)