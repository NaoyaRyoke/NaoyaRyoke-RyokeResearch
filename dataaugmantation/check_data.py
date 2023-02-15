import os, sys, cv2, glob, pickle, urllib
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils as utils
import torch.nn.init as init
import torch.utils.data as data
import torchvision
from torch.utils.data import Dataset
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.autograd import Variable
from PIL import Image
from itertools import chain
from skimage.io import imread

Height = 224 # 画像の高さ(このサイズにリサイズされる)
Width = 224 # 画像の幅(このサイズにリサイズされる)
modelPath = ""
moviePath = ""
imgPaths = ""
transform = transforms.Compose([
	# transforms.Resize((Height, Width), interpolation=transforms.InterpolationMode.BICUBIC),
	transforms.ToTensor(),
	transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
	])

# VGG16で転移学習
class VGG():
	def __init__(self):
		self.model = models.vgg16_bn(weights=models.VGG16_BN_Weights.IMAGENET1K_V1)
		self.transformClassifier()
		self.transGrad(True)
		
	# 全結合層の変形(最終出力層を学習クラス数に変形する必要があるため)
	def transformClassifier(self):
		self.model.classifier = nn.Sequential(
			nn.Linear(512 * 7 * 7, 4096),
			nn.ReLU(True),
			nn.Dropout(),
			nn.Linear(4096, 4096),
			nn.ReLU(True),
			nn.Dropout(),
			nn.Linear(4096, 10)
			)
	# 微分するかしないかみたいなやつ
	def transGrad(self, Boo):
		for p in self.model.features.parameters():
			p.requires_grad = Boo


# MobileNetで転移学習
class MobileNet():
	def __init__(self):
		self.model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V2)
		self.transformClassifier()
		self.transGrad(True)
		
	# 全結合層の変形(最終出力層を学習クラス数に変形する必要があるため)
	def transformClassifier(self):
		self.model.classifier = nn.Sequential(
			nn.Dropout(p=0.2, inplace=False),
			nn.Linear(1280, 10)
			)
	# 微分するかしないかみたいなやつ
	def transGrad(self, Boo):
		for p in self.model.features.parameters():
			p.requires_grad = Boo

# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture(moviePath)

testData = list(chain.from_iterable([glob.glob('{}/*'.format(d)) for d in glob.glob('{}/*'.format(imgPaths))]))

# model = VGG()
model = MobileNet()
model = model.model
model.load_state_dict(torch.load(modelPath))

for imgPath in testData:
	img = cv2.imread(imgPath, cv2.COLOR_BGR2RGB)
	image = transform(image).unsqueeze(dim=0)
	output = model(image).max(1)[1].item()


while True:
	ret, frame = cap.read()
	if frame is None:
		continue
	
	frame = cv2.rotate(frame, cv2.ROTATE_180)
	frame = cv2.resize(frame, (224,224), interpolation=cv2.INTER_CUBIC)
	image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	# image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
	image = transform(image).unsqueeze(dim=0)
	output = model(image).max(1)[1].item()
	# print(output)
	# print(type(frame), frame.shape)

	label = f"{output}"
	cv2.putText(frame, label, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
	
	cv2.imshow("camera", frame)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()