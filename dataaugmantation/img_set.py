import cv2, glob, os, shutil

imgsPath = r""
trainPath = r""
trainDatasetPathName = os.path.basename(imgsPath)
classesPath =[os.path.basename(i) for i in glob.glob(imgsPath+"\*")]

for classPath in classesPath:
	index = 0
	loadPath = os.path.join(imgsPath, classPath)
	savePath = os.path.join(trainPath, trainDatasetPathName, classPath)
	inFolderPaths = glob.glob(loadPath+"\*")
	for inFolderPath in inFolderPaths:
		inFolderPathName = os.path.basename(inFolderPath)
		# print(inFolderPathName)
		for inFolderImgPath in glob.glob(inFolderPath+"\*"):
			# print(inFolderImgPath)
			imgExt = os.path.splitext(inFolderImgPath)[1]
			# print(inFolderImgPath, savePath, str(index), imgExt)
			shutil.copyfile(inFolderImgPath, os.path.join(savePath, str(index)+imgExt))
			index += 1
