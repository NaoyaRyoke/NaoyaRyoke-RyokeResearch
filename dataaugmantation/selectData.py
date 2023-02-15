import cv2, glob, os, shutil
import numpy as np

trainPath = r""
tempPath = r""
trainDatasetPathName = os.path.basename(trainPath)
classesPath =[os.path.basename(i) for i in glob.glob(trainPath+"\*")]

for classPath in classesPath:
	images = np.random.permutation(glob.glob(os.path.join(trainPath, classPath)+"/*"))
	savePath = os.path.join(tempPath, classPath)
	for i in range(0, 500):
		img = images[i%len(images)]
		imgExt = os.path.splitext(img)[1]
		shutil.copyfile(img, os.path.join(savePath, str(i)+imgExt))
	# os.makedirs(os.path.join(tempPath, classPath))
