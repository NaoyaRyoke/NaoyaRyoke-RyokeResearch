import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils as utils
import torch.nn.init as init
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.autograd import Variable
from PIL import Image
import glob
import pickle
import urllib, cv2
from itertools import chain
from skimage.io import imread

EpochNum = 100 # 学習回数
Height = 224 # 画像の高さ(このサイズにリサイズされる)
Width = 224 # 画像の幅(このサイズにリサイズされる)
BatchSize = 10 # (ミニバッチ数)
trainPath = r"" # 学習用
valPath = r"" # 推論テスト用
modelPath = r"" # モデルを保存するパス
onnxPath = r""

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

# DataLoaderには画像のパスがデフォルトで実装されていないため追加
class VGGDataset(torch.utils.data.Dataset):
	def __init__(self, imgPaths, transform=None):
		self.transform = transform
		self.ImagePaths = list(chain.from_iterable([glob.glob('{}/*'.format(d)) for d in glob.glob('{}/*'.format(imgPaths))]))
		# print(len(self.ImagePaths))
		self.LabelName = [os.path.basename(fn) for fn in glob.glob('{}/*'.format(imgPaths))]
		self.Labels = [self.LabelName.index(l) for l in [os.path.basename(os.path.dirname(fn)) for fn in self.ImagePaths]]

	def __getitem__(self, index):
		image = None
		imgPath = self.ImagePaths[index]
		label = self.Labels[index]

		# with open(imgPath, 'rb') as f:
		# 	image = Image.open(f)
		# 	image = image.convert('RGB')
		image = cv2.imread(imgPath)
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		image = cv2.resize(image, dsize=(Height, Width), interpolation=cv2.INTER_CUBIC)

		if self.transform:
			image = self.transform(image)
		return image, label, imgPath

	def __len__(self):
		return len(self.ImagePaths)

# 学習するところ
class Trainer():
	def __init__(self, model, optimizer, criterion, trainLoader, valLoader, transform):
		self.model = model
		self.optimizer = optimizer
		self.criterion = criterion
		self.trainLoader = trainLoader
		self.valLoader = valLoader
		self.transform = transform
	
	def TrainEpoch(self, epoch):
		# 学習
		self.model.train()
		totalTrainLoss = 0 
		trainCorrect = 0 
		trainStr = "" 
		# print(len(self.trainLoader))
		for batchIdx, (data, target, imgData) in enumerate(self.trainLoader):
			data, target = Variable(data.cuda()), Variable(target.cuda()) # 学習用データをcudaに格納
			output = self.model(data) # 
			loss = self.criterion(output, target)
			totalTrainLoss += loss.data.item()
			self.optimizer.zero_grad()
			loss.backward()
			self.optimizer.step()

			pred = output.data.max(dim=1)[1]
			trainCorrect += pred.eq(target.data).cpu().sum()
			trainStr = "{}, epoch : {:3} train_loss : {:3.5} train_acc : {:3.5}".format(str(batchIdx), str(epoch+1), str(totalTrainLoss/((batchIdx+1)*BatchSize)), str(100*trainCorrect.data.item() / ((batchIdx+1)*BatchSize)))
			print("\r"+trainStr, end="")
		# 推論
		self.model.eval()
		totalValLoss = 0
		valCorrect = 0
		valStr = ""
		with torch.no_grad():
			for batchIdx, (data, target, imgData) in enumerate(self.valLoader):
				data, target = Variable(data.cuda(), requires_grad=False), Variable(target.cuda())
				output = self.model(data)
				totalValLoss += self.criterion(output, target).data.item()
				pred = output.data.max(dim=1)[1]
				valCorrect += pred.eq(target.data).cpu().sum()
				valStr = trainStr + " val_loss : {:3.5} val_acc : {:3.5}".format(str(totalValLoss/((batchIdx+1)*BatchSize)), str(100*valCorrect.data.item() / ((batchIdx+1)*BatchSize)))
				print("\r"+valStr, end="")
		print()
	
	def main(self):
		for epoch in range(EpochNum):
			self.TrainEpoch(epoch)
			with open(modelPath.format(str(epoch)), "wb") as f:
				torch.save(self.model.state_dict(), f)
		inputData = torch.autograd.Variable(torch.randn(1, 3, 224, 224), requires_grad=True)
		torch.onnx.export(
			self.model.to('cpu'), # model being run
			inputData,  # model input (or a tuple for multiple inputs)
			onnxPath, # where to save the model (can be a file or file-like object)
			export_params=True, # store the trained parameter weights inside the model file
			opset_version=10, # the ONNX version to export the model to
			do_constant_folding=True, # whether to execute constant folding for optimization
			input_names = ['input'], # the model's input names
			output_names = ['output'], # the model's output names
		)
		

# 前準備
if __name__ == '__main__':
	transform = transforms.Compose([
		# transforms.Resize((Height, Width), interpolation=transforms.InterpolationMode.BICUBIC),
		transforms.ToTensor(),
		transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
		])
	# VGG = VGG()
	VGG = MobileNet()
	model = VGG.model.cuda()
	optimizer = optim.SGD(model.classifier.parameters(), lr=0.00001, momentum=0.9)
	# criterion = nn.CrossEntropyLoss()
	criterion = nn.MultiMarginLoss()
	criterion.cuda()
	# trainLoader = datasets.ImageFolder(root=trainPath, transform=transform)
	trainLoader = VGGDataset(trainPath, transform=transform)
	# print(len(trainLoader))
	trainLoader = torch.utils.data.DataLoader(trainLoader, batch_size=BatchSize, shuffle=True, num_workers=1, pin_memory=True)
	# valLoader = datasets.ImageFolder(root=valPath, transform=transform)
	valLoader = VGGDataset(valPath, transform=transform)
	# valLoader = VGGDataset(valPath, transform=transform)
	valLoader = torch.utils.data.DataLoader(valLoader, batch_size=BatchSize, shuffle=False, num_workers=1, pin_memory=True)

	train = Trainer(model, optimizer, criterion, trainLoader, valLoader, transform)
	train.main()
