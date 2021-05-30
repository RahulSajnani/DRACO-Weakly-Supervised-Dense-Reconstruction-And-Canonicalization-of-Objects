import sys, os
sys.path.append(os.path.abspath(os.path.join('./models')))
sys.path.append(os.path.abspath(os.path.join('./Data_Loaders')))
sys.path.append(os.path.abspath(os.path.join('./Loss_Functions')))
from arch import SegNet, SegNet_Split
from arch_2 import Depth_Net
import data_loader
import torch.nn as nn
import argparse
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torchvision import transforms, utils
import config
import torch
import tqdm
import loss_functions
import gc


def init_weights(m):
    if type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
        #print('xavier')

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Training parameters')
    
    parser.add_argument('--dataset', help = 'Path to dataset', required=True)
    parser.add_argument('--learning_rate', help = 'Learning rate', default = config.LEARNING_RATE)
    
    args = parser.parse_args()    

    num_gpus = torch.cuda.device_count()
    train_data_set = data_loader.MultiView_dataset(args.dataset, train = 2, transform = data_loader.MultiView_dataset.ToTensor())
    train_dataloader = DataLoader(train_data_set, batch_size = config.BATCH_SIZE, num_workers = 2* num_gpus, shuffle=True)
    
    val_data_set = data_loader.MultiView_dataset(args.dataset, transform = data_loader.MultiView_dataset.ToTensor(), train = 0)
    val_dataloader = DataLoader(val_data_set, batch_size = 1, shuffle=True)
   
    depth_network = SegNet(pretrained=True)
    #depth_network = SegNet_Split()
    #depth_network = Depth_Net()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs!")
        depth_network = nn.DataParallel(depth_network)
    
    depth_network.to(device)

    print("GPU Metrics")
    print("-"*11)
    if torch.cuda.device_count():
        for i in range(torch.cuda.device_count()):
            print(torch.cuda.get_device_name(i))
            print('Memory Usage:')
            print('Allocated:', round(torch.cuda.memory_allocated(i)/1024**3,1), 'GB')
            print('Cached:   ', round(torch.cuda.memory_cached(i)/1024**3,1), 'GB')
    #print()

    print(num_gpus, "gpus in use", "\n lr:")
    print(float(args.learning_rate))
    optimizer = optim.Adam(depth_network.parameters(), float(args.learning_rate))
    depth_network = depth_network.float()
    num_samples = len(train_dataloader.dataset)/ config.BATCH_SIZE

    #scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[ 10, 20, 30, 50, 60], gamma=0.1) 
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[ 10, 30, 50, 70, 80, 90], gamma=0.1) 
    criterion = torch.nn.BCELoss()
    criterion_2 = loss_functions.Photometric_loss()
    criterion_3 = loss_functions.Smoothness_loss()
    #criterion_4 = loss_functions.Geometric_loss()
    # 2.5 1.5 0.08 works fine.
    w_bce = 2.0
    w_photo = 1.0
    w_smooth = 0.1
    train_loss = []
    val_loss = []
    train_acc = []
    val_acc = []
    #depth_network.apply(init_weights)
    for epoch in range(config.EPOCHS):
        running_loss_train = 0.0
        # training script
        mvg_loss = 0
        mask_loss = 0
        depth_network.train()
        progress = tqdm.tqdm(range(len(train_dataloader)), f"Epoch {epoch}", unit="", unit_scale=True, position = 0)
        for batch, data_sample in enumerate(train_dataloader):


            if torch.cuda.is_available():
                #data_sample['masks'] = data_sample['masks'].to(device).float()
                data_mask_ref = data_sample['masks'][:, 0].to(device).float()
                data_sample['views'] = data_sample['views'].to(device).float()
                #data_sample['poses'] = data_sample['poses'].to(device).float()
                #data_sample['intrinsics'] = data_sample['intrinsics'].to(device).float()
                # labels = labels.cuda()

            # print(data_input.shape)
            optimizer.zero_grad()
            #output = depth_network(data_sample['views'][:, 0])

            #tr_op = [output[0]]

            #for i in range(1, data_sample['num_views'][0]):
            #    tr_op.append(depth_network(data_sample['views'][:, i])[0]) # [B, 1, H, W]
            #    
            #tr_op = torch.cat(tr_op, dim=1)

            ## Send the output to the devices
            #output[1].to(device).float()
            #output[0].to(device).float()
            #tr_op.to(device).float()
            #
            ## print(output[1].size())
            ## print(data_mask.size())
            #loss = criterion(output[1], data_mask_ref)
            #loss_2 = criterion_2(data_sample, output[0])
            #loss_3 = criterion_3(output[0], data_sample)
            #loss_4 = criterion_4(data_sample, tr_op)
            #mvg_loss += loss_2 
            output = depth_network(data_sample['views'][:, 0])

            ## Send the output to the devices
            output[1].to(device).float()
            output[0].to(device).float()

            #print(output[1].size())
            # print(data_mask.size())
            #loss = criterion(output[1], data_mask_ref)
            loss_2 = criterion_2(data_sample, output[0])
            loss_3 = criterion_3(output[0], data_sample)
            #mask_loss += loss
            mvg_loss += loss_2
            #print(loss_2)
            # 1.5 0.3 0.9
            loss_total =  w_photo*loss_2.to(device) + w_smooth*loss_3.to(device)# + loss_4
            #print(loss_total)
            running_loss_train += loss_total
            loss_total.backward()
            progress.update()
            optimizer.step()
            torch.cuda.empty_cache()
            gc.collect()
        print('\n Training Loss: ', running_loss_train / num_samples)
        print('\n MVG LOSS', mvg_loss/num_samples)
        #print('\n Mask LOSS', mask_loss/num_samples)
        
	# depth_network.eval()
        scheduler.step()
        # running_loss_val = 0.0

        # for batch, data_sample in enumerate(val_dataloader):

        #     data_input = data_sample['view_0']

        #     if use_cuda and torch.cuda.is_available():
        #         data_input = data_input.cuda()
        #         data_mask = data_sample['mask_0'].cuda()
        #         # labels = labels.cuda()

        #     # print(data_input.shape)

        #     output = depth_network(data_input.float())
        #     # print(output[1].size())
        #     # print(data_mask.size())
        #     loss = criterion(output[1], data_mask.float())
        #     running_loss_val += loss.item()
        # print('\nVal Loss: ', running_loss_val / len(val_dataloader.dataset
    #torch.save(depth_network.state_dict(), '../checkpoint.pth')
        if (epoch+1)%30==0:
            file_name = f"../check_disp/overfit_100_{epoch+1}_net_depth_only_{w_bce}_{w_photo}_{w_smooth}__tgt_mask.pth"
            print('checkpoint saved in ',file_name)
            if num_gpus > 1:
                torch.save(depth_network.module.state_dict(), file_name)
            else:
                torch.save(depth_network.state_dict(), file_name)
    #torch.save(depth_network.module.state_dict(), '../check_disp/final_test.pth')
