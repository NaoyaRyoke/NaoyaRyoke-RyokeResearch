import numpy as np
import glob
import os
import xml.etree.ElementTree as ET
from xml.dom import minidom
import yaml
import random
import cv2
from PIL import Image, ImageDraw

def main():
	bgImagePath = r""
	nuisanceAnimalImagesPath = r""
	resultDatasetPath = r""
	startName = 1
	makeImageLength = 10000
	changeInterval = 1000
	baseSize = 224
	ConfusionImages(bgImagePath, nuisanceAnimalImagesPath, resultDatasetPath, changeInterval, baseSize).main(startName, makeImageLength)
# End_Function

class ConfusionImages():
	"""
	@parm
	_bgImagePath[str]:selectArea.pyで生成したデータセットを指定
	_nuisanceAnimalImagesPath[str]:透過した害獣の画像
	_resultDatasetPath[str]:生成したデータを保存するパス(PascalVOC方式)
	_className[str]:生成する画像のクラス
	"""
	def __init__(self, _bgImagePath, _nuisanceAnimalImagesPath, _resultDatasetPath, _changeInterval, _baseSize):
		self.bgImagePath = _bgImagePath
		self.bgImages = np.random.permutation(glob.glob(self.bgImagePath+"/*"))
		self.nuisanceAnimalImagesPath = _nuisanceAnimalImagesPath
		self.nuisanceAnimalImages = np.random.permutation(glob.glob(self.nuisanceAnimalImagesPath+"/*"))
		self.resultDatasetPath = _resultDatasetPath
		self.changeInterval = _changeInterval
		self.baseSize = _baseSize
	# End_Initialize

	"""
	画像重ね合わせるところ
	@parm
	_areaPath[str]:createAreaにあるymlファイルのパス
	_nuisanceImagePath[str]:重ねる透過害獣画像のパス
	_fileName[str]:生成する画像とymlの名前
	"""
	def ConfusionImage(self, _bgPath, _nuisanceImagePath, _fileName, _imgSize):
		bgImagePath = _bgPath
		nuisanceImagePath = _nuisanceImagePath
		fileName = _fileName
		imgSize = _imgSize

		# 害獣の画像の編集
		nuisanceSize = int(min(imgSize, imgSize*random.randrange(85, 98, 2)*0.01))
		nuisanceImg = cv2.imread(nuisanceImagePath, cv2.IMREAD_UNCHANGED)
		nuisanceImg = cv2.resize(nuisanceImg, (int(nuisanceSize), int((nuisanceImg.shape[0]*nuisanceSize)/nuisanceImg.shape[1])))
		if nuisanceImg.shape[0] > nuisanceSize:
			nuisanceImg = cv2.resize(nuisanceImg, (int((nuisanceImg.shape[1]*nuisanceSize)/nuisanceImg.shape[0]), int(nuisanceSize)))
		nuisanceImg = cv2.cvtColor(nuisanceImg, cv2.COLOR_BGRA2RGBA)
		# 重ねるためにcv2->Imageにする必要がある
		nuisanceImg = Image.fromarray(nuisanceImg)
		nuisanceImg = nuisanceImg.convert('RGBA')
		
		# 画像を貼り付ける場所を決める
		Point = (min(random.randint(0,10), imgSize-nuisanceImg.size[0]), min(random.randint(0,10), imgSize-nuisanceImg.size[1]))

		# 背景画像の編集(害獣と一緒)
		bgImg = cv2.imread(bgImagePath, cv2.IMREAD_COLOR)
		bgImg = cv2.resize(bgImg, (imgSize, imgSize))
		bgImg = cv2.cvtColor(bgImg, cv2.COLOR_BGR2RGB)
		bgImg = Image.fromarray(bgImg)
		bgImg = bgImg.convert('RGBA')

		# 同じサイズの画角
		frameImage = Image.new('RGBA', bgImg.size, (255, 255, 255, 0))

		# 座標を指定して重ねる
		frameImage.paste(nuisanceImg, Point, nuisanceImg)
		resultImg = Image.alpha_composite(bgImg, frameImage)

		# OpenCV形式画像へ変換
		resultImg = cv2.cvtColor(np.asarray(resultImg), cv2.COLOR_BGRA2RGBA)

		# 書き込み
		cv2.imwrite(self.resultDatasetPath+"//"+fileName+".jpg", resultImg)
	# End_Function

	"""
	main文
	@parm
	_startName[int]:連番の最初の数字
	_maxImageLength[int]:画像生成数
	"""
	def main(self, _startName, _maxImageLength):
		startName = _startName
		maxImageLength = _maxImageLength
		# print(self.nuisanceAnimalImages)
		for i in range(startName, (startName+maxImageLength)):
			self.ConfusionImage(self.bgImages[i%len(self.bgImages)], self.nuisanceAnimalImages[i%len(self.nuisanceAnimalImages)], str(i).zfill(8), int(self.baseSize-((i-startName)/self.changeInterval)*12))
	# End_Function
# End_Class

if __name__ == '__main__':
	main()