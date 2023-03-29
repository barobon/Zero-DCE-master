import torch
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn
import torch.optim
import os
import sys
import argparse
import time
import dataloader
import model
import Myloss
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

#loss
def plot_loss(train_loss_list, val_loss_list):
	plt.plot(train_loss_list, label='Training loss')
	plt.plot(val_loss_list, label='Validation loss')
	plt.xlabel('Epoch')
	plt.ylabel('Loss')
	plt.legend()
	plt.savefig(config.snapshots_folder+'loss_graph.png', dpi=800)

def train(config):

	os.environ['CUDA_VISIBLE_DEVICES']='0'

	DCE_net = model.enhance_net_nopool().cuda()

	DCE_net.apply(weights_init)
	if config.load_pretrain == True:
	    DCE_net.load_state_dict(torch.load(config.pretrain_dir))
	train_dataset = dataloader.lowlight_loader(config.lowlight_images_path)
	train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=True, num_workers=config.num_workers, pin_memory=True)

	val_dataset = dataloader.lowlight_loader(config.val_path)
	val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config.val_batch_size, shuffle=False, num_workers=config.num_workers, pin_memory=True)



	L_color = Myloss.L_color()
	L_spa = Myloss.L_spa()

	L_exp = Myloss.L_exp(16,0.6)
	L_TV = Myloss.L_TV()


	optimizer = torch.optim.Adam(DCE_net.parameters(), lr=config.lr, weight_decay=config.weight_decay)
	
	DCE_net.train()

	train_loss_list = [] #loss 값을 저장하는 리스트
	val_loss_list = []
	lowest_loss = 999999
	lowest_val_loss = 999999

	for epoch in range(config.num_epochs):
		train_loss = 0
		val_loss = 0
		for iteration, img_lowlight in enumerate(train_loader):

			img_lowlight = img_lowlight.cuda()

			enhanced_image_1,enhanced_image,A  = DCE_net(img_lowlight)

			Loss_TV = 200*L_TV(A)
			
			loss_spa = torch.mean(L_spa(enhanced_image, img_lowlight))

			loss_col = 5*torch.mean(L_color(enhanced_image))

			loss_exp = 10*torch.mean(L_exp(enhanced_image))
			
			
			# best_loss
			loss =  Loss_TV + loss_spa + loss_col + loss_exp
			train_loss += loss.item()
			#

			
			optimizer.zero_grad()
			loss.backward()
			torch.nn.utils.clip_grad_norm(DCE_net.parameters(),config.grad_clip_norm)
			optimizer.step()

			if ((iteration+1) % config.display_iter) == 0:
				print("Loss at iteration", iteration+1, ":", loss.item())
			if ((iteration+1) % config.snapshot_iter) == 0:
				torch.save(DCE_net.state_dict(), config.snapshots_folder + "Epoch" + str(epoch) + '.pth')

		epoch_train_loss = train_loss / len(train_loader)	# 손실함수의 누적값을 배치 크기로 나누어 각 epoch에서의 손실함수 값을 구한다다
		train_loss_list.append(epoch_train_loss)  # 각 epoch에서의 loss 값을 리스트에 추가

		if (epoch_train_loss < lowest_loss):
			lowest_loss = epoch_train_loss
			torch.save(DCE_net.state_dict(), config.snapshots_folder + 'train_lowest.pth')


		val_loss = 0
		with torch.no_grad():
			DCE_net.eval()  # set the network to evaluation mode
			for iteration, img_lowlight_val in enumerate(val_loader):
				img_lowlight_val = img_lowlight_val.cuda()

				enhanced_image_1_val, enhanced_image_val, A_val = DCE_net(img_lowlight_val)

				Loss_TV_val = 200 * L_TV(A_val)
				loss_spa_val = torch.mean(L_spa(enhanced_image_val, img_lowlight_val))
				loss_col_val = 5 * torch.mean(L_color(enhanced_image_val))
				loss_exp_val = 10 * torch.mean(L_exp(enhanced_image_val))

				loss_val = Loss_TV_val + loss_spa_val + loss_col_val + loss_exp_val
				val_loss += loss_val.item()

		val_loss /= len(val_loader)
		val_loss_list.append(val_loss)
		print("Epoch:", epoch, "Training loss:", epoch_train_loss, "Validation loss:", val_loss)

		if (val_loss < lowest_val_loss):
			lowest_val_loss = val_loss
			torch.save(DCE_net.state_dict(), config.snapshots_folder + 'val_lowest.pth')
		DCE_net.train()

	plot_loss(train_loss_list, val_loss_list)





if __name__ == "__main__":

	parser = argparse.ArgumentParser()

	# Input Parameters
	parser.add_argument('--lowlight_images_path', type=str, default="data/train_data/")
	parser.add_argument('--val_path', type=str, default="data/val_data/")
	parser.add_argument('--lr', type=float, default=0.0001)
	parser.add_argument('--weight_decay', type=float, default=0.0001)
	parser.add_argument('--grad_clip_norm', type=float, default=0.1)
	parser.add_argument('--num_epochs', type=int, default=200)
	parser.add_argument('--train_batch_size', type=int, default=8)
	parser.add_argument('--val_batch_size', type=int, default=4)
	parser.add_argument('--num_workers', type=int, default=4)
	parser.add_argument('--display_iter', type=int, default=10)
	parser.add_argument('--snapshot_iter', type=int, default=10)
	parser.add_argument('--snapshots_folder', type=str, default="snapshots/")
	parser.add_argument('--load_pretrain', type=bool, default= False)
	parser.add_argument('--pretrain_dir', type=str, default= "snapshots/Epoch99.pth")

	config = parser.parse_args()

	if not os.path.exists(config.snapshots_folder):
		os.mkdir(config.snapshots_folder)


	train(config)








	
