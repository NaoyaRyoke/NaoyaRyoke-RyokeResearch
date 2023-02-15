import glob
import os
import cv2

oriPath = r""
resultPath = r""

oriGlob = glob.glob(oriPath+"//*")
# print(os.path.basename(oriGlob[0]))

for i in oriGlob:
	img = cv2.imread(i)
	# dust = cv2.resize(img, (224,224))
	dust = cv2.flip(img, 1)
	cv2.imwrite(os.path.join(resultPath,os.path.basename(i)), dust)