import cv2
import numpy as np
import glob
import os

def main():
	kusa = glob.glob(r".\\*")
	print(os.path.splitext(os.path.basename(kusa[0]))[0])
	deleteBack.main(r"image", r"result\\")

class deleteBack:
	def main(imagePath, resultPath):
		deleteBack.deleteBackImages(imagePath, resultPath)

	def deleteBackImages(imagePath, resultPath):
		for image in glob.glob(imagePath+"\*"):
			deleteBack.deleteBack(image, resultPath+os.path.splitext(os.path.basename(image))[0]+"_result.png")

	def deleteBack(imgPath, resultPath):
		img = cv2.imread(imgPath, -1)
		img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
		color_lower = np.array([0, 0, 150, 255])
		color_upper = np.array([100, 100, 255, 255])
		img_mask = cv2.inRange(img, color_lower, color_upper)
		img_bool = cv2.bitwise_not(img, img, mask=img_mask)
		cv2.imwrite(resultPath, img_bool)

if __name__ == '__main__':
	main()