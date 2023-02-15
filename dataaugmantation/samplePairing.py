import numpy as np
import glob
import os
import xml.etree.ElementTree as ET
from xml.dom import minidom
import yaml
import random
import cv2
import math
import random
from PIL import Image, ImageDraw

def main():
	dataset = r""
	resultPath = r""
	classLabel = os.listdir(dataset)
	maximum = 14000
	for i in range(len(classLabel)):
		resultClassPath = os.path.join(resultPath, classLabel[i])
		os.mkdir(resultClassPath)
		maxImageLength = maximum
		SamplePairing().SamplePairing(dataset, i, 224, 0, maxImageLength, resultClassPath)

class SamplePairing():
	def __init__(self):
		pass
	
	### targetDataset, otherDataset
	def setDataset(self, _dataset, _targetClass):
		dataset = glob.glob(_dataset+"\\*")
		targetDataset = glob.glob(dataset[_targetClass]+"\\*")
		otherDataset = []
		for i, datas in enumerate(dataset):
			if i==_targetClass:	continue
			otherDataset+=glob.glob(datas+"\\*")
		return np.random.permutation(targetDataset), np.random.permutation(otherDataset)
	
	def SamplePairing(self, _datasets, _targetClass, _imgSize, _start, _maxImageLength, _resultPath):
		targetDataset, otherDataset = self.setDataset(_datasets, _targetClass)
		imgSize = _imgSize
		maxImageLength = _maxImageLength
		startName = _start
		resultPath = _resultPath

		for i in range(0, maxImageLength):
			labelAImg = cv2.imread(targetDataset[i%len(targetDataset)], cv2.IMREAD_UNCHANGED)
			labelAImg = cv2.resize(labelAImg, (imgSize, imgSize))

			labelBImg = cv2.imread(otherDataset[i%len(otherDataset)], cv2.IMREAD_UNCHANGED)
			labelBImg = cv2.resize(labelBImg, (imgSize, imgSize))

			blended = cv2.addWeighted(labelAImg, 0.5, labelBImg, 0.5, 0)

			# # cv2.imshow("unpi", blended)
			cv2.imwrite(os.path.join(resultPath,str(startName+i).zfill(6)+".jpg"), blended)
			# cv2.waitKey(0)
		return


if __name__ == '__main__':
	main()