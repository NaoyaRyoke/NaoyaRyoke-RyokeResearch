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

# 生成したい合計を指定して，それを手法数で割る
# それをクラス回繰り返す 

def main():
	dataset = r""
	resultPath = r""
	classLabel = os.listdir(dataset)
	maximum = 14000
	print(classLabel)
	for i in range(len(classLabel)):
		resultClassPath = os.path.join(resultPath, classLabel[i])
		os.mkdir(resultClassPath)
		# maxImageLength = int(maximum / 15)
		# addImageLength = (maximum%15)

		# maxImageLength = int(maximum / 2)
		# addImageLength = (maximum%2)

		maxImageLength = int(maximum / 13)
		addImageLength = (maximum%13)

		# Linear Methods
		# MixingImages().Mixup(dataset, i, 224, 0, maxImageLength, resultClassPath)
		# MixingImages().Bcplus(dataset, i, 224, maxImageLength, maxImageLength, resultClassPath)
		# Non Linear Methods
		MixingImages().VerticalConcat(dataset, i, 224, maxImageLength*2, maxImageLength, resultClassPath)
		MixingImages().HorizontalConcat(dataset, i, 224, maxImageLength*3, maxImageLength, resultClassPath)
		MixingImages().MixedConcat(dataset, i, 224, maxImageLength*4, maxImageLength, resultClassPath)
		MixingImages().Random2x2(dataset, i, 224, maxImageLength*5, maxImageLength, resultClassPath)
		MixingImages().VHMixup(dataset, i, 224, maxImageLength*6, maxImageLength, resultClassPath)
		MixingImages().RandomSquare(dataset, i, 224, maxImageLength*7, maxImageLength, resultClassPath)
		MixingImages().RandomRowInterval(dataset, i, 224, maxImageLength*8, maxImageLength, resultClassPath)
		MixingImages().RandomColumnInterval(dataset, i, 224, maxImageLength*9, maxImageLength, resultClassPath)
		MixingImages().RandomRow(dataset, i, 224, maxImageLength*10, maxImageLength, resultClassPath)
		MixingImages().RandomColumn(dataset, i, 224, maxImageLength*11, maxImageLength, resultClassPath)
		MixingImages().RandomPixels(dataset, i, 224, maxImageLength*12, maxImageLength, resultClassPath)
		MixingImages().RandomElements(dataset, i, 224, maxImageLength*13, maxImageLength, resultClassPath)
		MixingImages().NoisyMixup(dataset, i, 224, maxImageLength*14, maxImageLength+addImageLength, resultClassPath)
# End_Function

class MixingImages():
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
	
	def Mixup(self, _datasets, _targetClass, _imgSize, _start, _maxImageLength, _resultPath):
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

			mixCoeff = random.uniform(0.7,0.9)
			# mixCoeff = 0.7
			blended = cv2.addWeighted(labelAImg, mixCoeff, labelBImg, 1 - mixCoeff, 0)

			# cv2.imshow("unpi", blended)
			cv2.imwrite(os.path.join(resultPath,str(startName+i).zfill(6)+".jpg"), blended)
			cv2.waitKey(0)
		return

	def Bcplus(self, _datasets, _targetClass, _imgSize, _start, _maxImageLength, _resultPath):
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

			mixCoeff = random.uniform(0.4,0.8)
			# mixCoeff = 0.7
			labelAmean = labelAImg.mean()
			labelAStd = np.std(labelAImg)
			labelBmean = labelBImg.mean()
			labelBStd = np.std(labelBImg)

			p = 1. / (1. + (labelAStd/labelBStd) * (1. - mixCoeff) / mixCoeff)
			dst = (p*(labelAImg-labelAmean)+(1.-p)*(labelBImg-labelAmean)) / math.sqrt(math.pow(p,2)+math.pow(1-p,2))

			# cv2.imshow("unpi", dst)
			cv2.imwrite(os.path.join(resultPath,str(startName+i).zfill(6)+".jpg"), dst)
			cv2.waitKey(0)
		return

	def VerticalConcat(self, _datasets, _targetClass, _imgSize, _start, _maxImageLength, _resultPath):
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

			mixCoeff = random.uniform(0.2,0.4)
			# mixCoeff = 0.3

			dst = labelAImg.copy()
			H, W = labelAImg.shape[:2]
			# dst[0:0, H:W*mixCoeff] = labelBImg[0:0, H:W*mixCoeff]
			dst[0:int(W*mixCoeff),0:H] = labelBImg[0:int(W*mixCoeff),0:H]

			# cv2.imshow("unpi", dst)
			cv2.imwrite(os.path.join(resultPath,str(startName+i).zfill(6)+".jpg"), dst)
			cv2.waitKey(0)
		return

	def HorizontalConcat(self, _datasets, _targetClass, _imgSize, _start, _maxImageLength, _resultPath):
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

			mixCoeff = random.uniform(0.2,0.4)
			# mixCoeff = 0.3

			dst = labelAImg.copy()
			H, W = labelAImg.shape[:2]
			# dst[0:0, H:W*mixCoeff] = labelBImg[0:0, H:W*mixCoeff]
			dst[0:W,0:int(H*mixCoeff)] = labelBImg[0:W,0:int(H*mixCoeff)]

			# cv2.imshow("unpi", dst)
			cv2.imwrite(os.path.join(resultPath,str(startName+i).zfill(6)+".jpg"), dst)
			cv2.waitKey(0)
		return

	def MixedConcat_1(self, _labelAImg, _labelBImg, _name, _resultPath):
		labelAImg = _labelAImg
		labelBImg = _labelBImg
		name = _name
		resultPath = _resultPath
		
		# mixCoeff = random.uniform(0.1,0.9)
		hMixCoeff = random.uniform(0.6,0.9)
		wMixCoeff = random.uniform(0.1,0.4)
		# hMixCoeff = 0.7
		# wMixCoeff = 0.3

		dst = labelAImg.copy()
		H, W = labelAImg.shape[:2]
		# dst[0:0, H:W*mixCoeff] = labelBImg[0:0, H:W*mixCoeff]
		dst[0:int(W*wMixCoeff),0:int(H*hMixCoeff)] = labelBImg[0:int(W*wMixCoeff),0:int(H*hMixCoeff)]
		dst[int(W*wMixCoeff):W, int(H*hMixCoeff):H] = labelBImg[int(W*wMixCoeff):W, int(H*hMixCoeff):H]

		# cv2.imshow("unpi", dst)
		cv2.imwrite(os.path.join(resultPath,str(name).zfill(6)+".jpg"), dst)
		cv2.waitKey(0)
		return
	
	def MixedConcat_2(self, _labelAImg, _labelBImg, _name, _resultPath):
		labelAImg = _labelAImg
		labelBImg = _labelBImg
		name = _name
		resultPath = _resultPath

		# mixCoeff = random.uniform(0.1,0.9)
		hMixCoeff = random.uniform(0.1,0.4)
		wMixCoeff = random.uniform(0.6,0.9)
		# hMixCoeff = 0.3
		# wMixCoeff = 0.7

		dst = labelAImg.copy()
		H, W = labelAImg.shape[:2]
		# dst[0:0, H:W*mixCoeff] = labelBImg[0:0, H:W*mixCoeff]
		dst[0:int(W*wMixCoeff),0:int(H*hMixCoeff)] = labelBImg[0:int(W*wMixCoeff),0:int(H*hMixCoeff)]
		dst[int(W*wMixCoeff):W, int(H*hMixCoeff):H] = labelBImg[int(W*wMixCoeff):W, int(H*hMixCoeff):H]

		# cv2.imshow("unpi", dst)
		cv2.imwrite(os.path.join(resultPath,str(name).zfill(6)+".jpg"), dst)
		cv2.waitKey(0)
		return
	
	def MixedConcat_3(self, _labelAImg, _labelBImg, _name, _resultPath):
		labelAImg = _labelAImg
		labelBImg = _labelBImg
		name = _name
		resultPath = _resultPath

		# mixCoeff = random.uniform(0.1,0.9)
		hMixCoeff = random.uniform(0.6,0.9)
		wMixCoeff = random.uniform(0.6,0.9)
		# hMixCoeff = 0.7
		# wMixCoeff = 0.7

		dst = labelBImg.copy()
		H, W = labelBImg.shape[:2]
		# dst[0:0, H:W*mixCoeff] = labelBImg[0:0, H:W*mixCoeff]
		dst[0:int(W*wMixCoeff),0:int(H*hMixCoeff)] = labelAImg[0:int(W*wMixCoeff),0:int(H*hMixCoeff)]
		dst[int(W*wMixCoeff):W, int(H*hMixCoeff):H] = labelAImg[int(W*wMixCoeff):W, int(H*hMixCoeff):H]

		# cv2.imshow("unpi", dst)
		cv2.imwrite(os.path.join(resultPath,str(name).zfill(6)+".jpg"), dst)
		cv2.waitKey(0)
		return

	def MixedConcat_4(self, _labelAImg, _labelBImg, _name, _resultPath):
		labelAImg = _labelAImg
		labelBImg = _labelBImg
		name = _name
		resultPath = _resultPath

		# mixCoeff = random.uniform(0.1,0.9)
		hMixCoeff = random.uniform(0.1,0.4)
		wMixCoeff = random.uniform(0.1,0.4)
		# hMixCoeff = 0.3
		# wMixCoeff = 0.3

		dst = labelBImg.copy()
		H, W = labelAImg.shape[:2]
		# dst[0:0, H:W*mixCoeff] = labelBImg[0:0, H:W*mixCoeff]
		dst[0:int(W*wMixCoeff),0:int(H*hMixCoeff)] = labelAImg[0:int(W*wMixCoeff),0:int(H*hMixCoeff)]
		dst[int(W*wMixCoeff):W, int(H*hMixCoeff):H] = labelAImg[int(W*wMixCoeff):W, int(H*hMixCoeff):H]

		# cv2.imshow("unpi", dst)
		cv2.imwrite(os.path.join(resultPath,str(name).zfill(6)+".jpg"), dst)
		cv2.waitKey(0)
		return

	def MixedConcat(self, _datasets, _targetClass, _imgSize, _start, _maxImageLength, _resultPath):
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

			pattern = random.randint(1, 4)
			if pattern == 1:self.MixedConcat_1(labelAImg, labelBImg, startName+i, resultPath)
			elif pattern == 2:self.MixedConcat_2(labelAImg, labelBImg, startName+i, resultPath)
			elif pattern == 3:self.MixedConcat_3(labelAImg, labelBImg, startName+i, resultPath)
			else :self.MixedConcat_4(labelAImg, labelBImg, startName+i, resultPath)
		return

	def Random2x2(self, _datasets, _targetClass, _imgSize, _start, _maxImageLength, _resultPath):
		targetDataset, otherDataset = self.setDataset(_datasets, _targetClass)
		imgSize = _imgSize
		maxImageLength = _maxImageLength
		startName = _start
		resultPath = _resultPath

		for i in range(0, maxImageLength):
			pattern = np.random.permutation([[0,0,0,1], [0,0,1,0], [0,0,1,1], [0,1,0,0], [0,1,0,1], [0,1,1,0], [0,1,1,1], [1,0,0,1], [1,0,1,0], [1,0,1,1], [1,1,0,0], [1,1,0,1], [1,1,1,0]])
			labelAImg = cv2.imread(targetDataset[i%len(targetDataset)], cv2.IMREAD_UNCHANGED)
			labelAImg = cv2.resize(labelAImg, (imgSize, imgSize))

			labelBImg = cv2.imread(otherDataset[i%len(otherDataset)], cv2.IMREAD_UNCHANGED)
			labelBImg = cv2.resize(labelBImg, (imgSize, imgSize))

			mixCoeff = random.uniform(0.6,0.9)
			# mixCoeff = 0.7
			blended = cv2.addWeighted(labelAImg, mixCoeff, labelBImg, 1 - mixCoeff, 0)

			H, W = labelAImg.shape[:2]
			# dst[0:0, H:W*mixCoeff] = labelBImg[0:0, H:W*mixCoeff]
			if pattern[0][0]==1:blended[0:int(W/2),0:int(H/2)] = labelAImg[0:int(W/2),0:int(H/2)]
			if pattern[0][1]==1:blended[int(W/2):W,0:int(H/2)] = labelAImg[int(W/2):W,0:int(H/2)]
			if pattern[0][2]==1:blended[0:int(W/2),int(H/2):H] = labelAImg[0:int(W/2),int(H/2):H]
			if pattern[0][3]==1:blended[int(W/2):W,int(H/2):H] = labelAImg[int(W/2):W,int(H/2):H]

			# cv2.imshow("unpi", blended)
			cv2.imwrite(os.path.join(resultPath,str(startName+i).zfill(6)+".jpg"), blended)
			cv2.waitKey(0)
		return

	def VHMixup_1(self, _labelAImg, _labelBImg, _name, _resultPath):
		labelAImg = _labelAImg
		labelBImg = _labelBImg
		name = _name
		resultPath = _resultPath

		mixCoeff = random.uniform(0.6,0.9)
		# mixCoeff = 0.7
		blended = cv2.addWeighted(labelAImg, mixCoeff, labelBImg, 1 - mixCoeff, 0)

		hMixCoeff = random.uniform(0.6,0.9)
		wMixCoeff = random.uniform(0.1,0.4)
		# hMixCoeff = 0.7
		# wMixCoeff = 0.3

		H, W = labelAImg.shape[:2]
		# dst[0:0, H:W*mixCoeff] = labelBImg[0:0, H:W*mixCoeff]
		blended[int(W*wMixCoeff):W,0:int(H*hMixCoeff)] = labelAImg[int(W*wMixCoeff):W,0:int(H*hMixCoeff)]
		blended[0:int(W*wMixCoeff),int(H*hMixCoeff):H] = labelBImg[0:int(W*wMixCoeff),int(H*hMixCoeff):H]

		# cv2.imshow("unpi", blended)
		cv2.imwrite(os.path.join(resultPath,str(name).zfill(6)+".jpg"), blended)
		cv2.waitKey(0)
		return
	
	def VHMixup_2(self, _labelAImg, _labelBImg, _name, _resultPath):
		labelAImg = _labelAImg
		labelBImg = _labelBImg
		name = _name
		resultPath = _resultPath

		mixCoeff = random.uniform(0.6,0.9)
		# mixCoeff = 0.7
		blended = cv2.addWeighted(labelAImg, mixCoeff, labelBImg, 1 - mixCoeff, 0)

		hMixCoeff = random.uniform(0.1,0.4)
		wMixCoeff = random.uniform(0.6,0.9)
		# hMixCoeff = 0.3
		# wMixCoeff = 0.7

		H, W = labelAImg.shape[:2]
		# dst[0:0, H:W*mixCoeff] = labelBImg[0:0, H:W*mixCoeff]
		blended[int(W*wMixCoeff):W,0:int(H*hMixCoeff)] = labelBImg[int(W*wMixCoeff):W,0:int(H*hMixCoeff)]
		blended[0:int(W*wMixCoeff),int(H*hMixCoeff):H] = labelAImg[0:int(W*wMixCoeff),int(H*hMixCoeff):H]

		# cv2.imshow("unpi", blended)
		cv2.imwrite(os.path.join(resultPath,str(name).zfill(6)+".jpg"), blended)
		cv2.waitKey(0)
		return

	def VHMixup_3(self, _labelAImg, _labelBImg, _name, _resultPath):
		labelAImg = _labelAImg
		labelBImg = _labelBImg
		name = _name
		resultPath = _resultPath

		mixCoeff = random.uniform(0.6,0.9)
		# mixCoeff = 0.7
		blended = cv2.addWeighted(labelAImg, mixCoeff, labelBImg, 1 - mixCoeff, 0)

		hMixCoeff = random.uniform(0.1,0.4)
		wMixCoeff = random.uniform(0.1,0.4)
		# hMixCoeff = 0.3
		# wMixCoeff = 0.3

		H, W = labelAImg.shape[:2]
		# dst[0:0, H:W*mixCoeff] = labelBImg[0:0, H:W*mixCoeff]
		blended[0:int(W*wMixCoeff),0:int(H*hMixCoeff)] = labelBImg[0:int(W*wMixCoeff),0:int(H*hMixCoeff)]
		blended[int(W*wMixCoeff),int(H*hMixCoeff):H] = labelAImg[int(W*wMixCoeff),int(H*hMixCoeff):H]
		
		# cv2.imshow("unpi", blended)
		cv2.imwrite(os.path.join(resultPath,str(name).zfill(6)+".jpg"), blended)
		cv2.waitKey(0)
		return

	def VHMixup_4(self, _labelAImg, _labelBImg, _name, _resultPath):
		labelAImg = _labelAImg
		labelBImg = _labelBImg
		name = _name
		resultPath = _resultPath
		
		mixCoeff = random.uniform(0.6,0.9)
		# mixCoeff = 0.7
		blended = cv2.addWeighted(labelAImg, mixCoeff, labelBImg, 1 - mixCoeff, 0)

		hMixCoeff = random.uniform(0.6,0.9)
		wMixCoeff = random.uniform(0.6,0.9)
		# hMixCoeff = 0.7
		# wMixCoeff = 0.7

		H, W = labelAImg.shape[:2]
		# dst[0:0, H:W*mixCoeff] = labelBImg[0:0, H:W*mixCoeff]
		blended[0:int(W*wMixCoeff),0:int(H*hMixCoeff)] = labelAImg[0:int(W*wMixCoeff),0:int(H*hMixCoeff)]
		blended[int(W*wMixCoeff):W,int(H*hMixCoeff):H] = labelBImg[int(W*wMixCoeff):W,int(H*hMixCoeff):H]

		# cv2.imshow("unpi", blended)
		cv2.imwrite(os.path.join(resultPath,str(name).zfill(6)+".jpg"), blended)
		cv2.waitKey(0)
		return

	def VHMixup(self, _datasets, _targetClass, _imgSize, _start, _maxImageLength, _resultPath):
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

			pattern = random.randint(1, 4)
			if pattern == 1:self.VHMixup_1(labelAImg, labelBImg, startName+i, resultPath)
			elif pattern == 2:self.VHMixup_2(labelAImg, labelBImg, startName+i, resultPath)
			elif pattern == 3:self.VHMixup_3(labelAImg, labelBImg, startName+i, resultPath)
			else :self.VHMixup_4(labelAImg, labelBImg, startName+i, resultPath)
		return

	def RandomSquare(self, _datasets, _targetClass, _imgSize, _start, _maxImageLength, _resultPath):
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

			minSize = random.uniform(0.3,0.4)
			maxSize = random.uniform(0.6,0.7)
			# minSize = 0.3
			# maxSize = 0.7
			minSize = int(imgSize*minSize)
			maxSize = int(imgSize*maxSize)
			
			newBImg = labelBImg[minSize:maxSize, minSize:maxSize]

			mixCoeff = random.uniform(0.1,0.6)
			# mixCoeff = 0.5
			blended = labelAImg.copy()

			H, W = labelAImg.shape[:2]
			# dst[0:0, H:W*mixCoeff] = labelBImg[0:0, H:W*mixCoeff]
			blended[int(W*mixCoeff):int(W*mixCoeff)+(maxSize-minSize), int(H*mixCoeff):int(W*mixCoeff)+(maxSize-minSize)] = newBImg

			# cv2.imshow("unpi", blended)
			cv2.imwrite(os.path.join(resultPath,str(startName+i).zfill(6)+".jpg"), blended)
			cv2.waitKey(0)
		return

	def RandomRowInterval(self, _datasets, _targetClass, _imgSize, _start, _maxImageLength, _resultPath):
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

			# x = random.randint(0,1)*0.5

			minSize = random.uniform(0.3,0.4)
			maxSize = random.uniform(0.5,0.6)
			# minSize = 0.2+x
			# maxSize = 0.4+x

			mixCoeff = random.uniform(0.1,0.6)
			# mixCoeff = 0.5
			blended = labelAImg.copy()

			H, W = labelAImg.shape[:2]
			# dst[0:0, H:W*mixCoeff] = labelBImg[0:0, H:W*mixCoeff]
			blended[int(H*minSize):int(H*maxSize), 0:W] = labelBImg[int(H*minSize):int(H*maxSize), 0:W]

			# cv2.imshow("unpi", blended)
			cv2.imwrite(os.path.join(resultPath,str(startName+i).zfill(6)+".jpg"), blended)
			cv2.waitKey(0)
		return
	
	def RandomColumnInterval(self, _datasets, _targetClass, _imgSize, _start, _maxImageLength, _resultPath):
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

			# x = random.randint(0,1)*0.5

			minSize = random.uniform(0.3,0.4)
			maxSize = random.uniform(0.5,0.6)

			mixCoeff = random.uniform(0.1,0.6)
			# mixCoeff = 0.5
			blended = labelAImg.copy()

			H, W = labelAImg.shape[:2]
			# dst[0:0, H:W*mixCoeff] = labelBImg[0:0, H:W*mixCoeff]
			blended[0:W, int(H*minSize):int(H*maxSize)] = labelBImg[0:W, int(H*minSize):int(H*maxSize)]

			# cv2.imshow("unpi", blended)
			cv2.imwrite(os.path.join(resultPath,str(startName+i).zfill(6)+".jpg"), blended)
			cv2.waitKey(0)
		return
	
	def RandomRow(self, _datasets, _targetClass, _imgSize, _start, _maxImageLength, _resultPath):
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

			count = 0
			blended = labelAImg.copy()
			
			for j in range(0, 224):
				x = random.randint(0,10)
				if x >= 9:
					count += 1
				H, W = labelAImg.shape[:2]
				if x >= 9 and count <= 80:
					blended[j:j+1, 0:W] = labelBImg[j:j+1, 0:W]

			# cv2.imshow("unpi", blended)
			cv2.imwrite(os.path.join(resultPath,str(startName+i).zfill(6)+".jpg"), blended)
			cv2.waitKey(0)
		return
	
	def RandomColumn(self, _datasets, _targetClass, _imgSize, _start, _maxImageLength, _resultPath):
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

			count = 0
			blended = labelAImg.copy()
			
			for j in range(0, 224):
				x = random.randint(0,10)
				if x >= 9:
					count += 1
				H, W = labelAImg.shape[:2]
				if x >= 9 and count <= 80:
					blended[0:W, j:j+1] = labelBImg[0:W, j:j+1]

			# cv2.imshow("unpi", blended)
			cv2.imwrite(os.path.join(resultPath,str(startName+i).zfill(6)+".jpg"), blended)
			cv2.waitKey(0)
		return

	def RandomPixels(self, _datasets, _targetClass, _imgSize, _start, _maxImageLength, _resultPath):
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

			blended = labelAImg.copy()
			
			for k in range(0, 224):
				count = 0
				for j in range(0, 224):
					x = random.randint(0,10)
					if x >= 9:
						count += 1
					H, W = labelAImg.shape[:2]
					if x >= 9 and count <= 80:
						blended[k:k+1, j:j+1] = labelBImg[k:k+1, j:j+1]

			# cv2.imshow("unpi", blended)
			cv2.imwrite(os.path.join(resultPath,str(startName+i).zfill(6)+".jpg"), blended)
			cv2.waitKey(0)
		return

	def RandomElements(self, _datasets, _targetClass, _imgSize, _start, _maxImageLength, _resultPath):
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

			blended = labelAImg.copy()
			
			for l in range(0, 224):
				for k in range(0, 3):
					count = 0
					for j in range(0, 224):
						x = random.randint(0,10)
						if x >= 9:
							count += 1
						H, W = labelAImg.shape[:2]
						if x >= 9 and count <= 80:
							blended[l][j][k] = labelBImg[l][j][k]
			# cv2.imshow("unpi", blended)
			cv2.imwrite(os.path.join(resultPath,str(startName+i).zfill(6)+".jpg"), blended)
			cv2.waitKey(0)
		return

	def NoisyMixup(self, _datasets, _targetClass, _imgSize, _start, _maxImageLength, _resultPath):
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

			mixCoeff = random.uniform(0.4,0.8)
			# mixCoeff = 0.7
			blended = cv2.addWeighted(labelAImg, mixCoeff, labelBImg, 1 - mixCoeff, 0)

			H, W, C = labelAImg.shape
			ptsX = np.random.randint(0, imgSize-1, 500)
			ptsY = np.random.randint(0, imgSize-1, 500)
			blended[ptsY, ptsX] = (255, 255, 255)

			ptsX = np.random.randint(0, imgSize-1, 500)
			ptsY = np.random.randint(0, imgSize-1, 500)
			blended[ptsY, ptsX] = (0, 0, 0)

			# cv2.imshow("unpi", blended)
			cv2.imwrite(os.path.join(resultPath,str(startName+i).zfill(6)+".jpg"), blended)
			cv2.waitKey(0)
		return

if __name__ == '__main__':
	main()