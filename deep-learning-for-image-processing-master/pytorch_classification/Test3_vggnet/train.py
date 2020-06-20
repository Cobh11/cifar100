# -*- coding:UTF-8 -*-
import torch.nn as nn
from torchvision import transforms, datasets
import json
import os
import torch.optim as optim
from model import vgg
import torch
from PIL import Image
from torch.autograd import Variable
#from torch.utils.data import Data
import torchvision
import matplotlib.pyplot as plt
import numpy as np

torch.manual_seed(1)
#hyper paras
batch_size = 18
#lr = 0.005
#DOWNLOAD_MNIST = False
#N_TEST_IMG = 5

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

train_transform =transforms.Compose([
                transforms.Resize([224,224]),  
                #transforms.RandomResizeCrop(224),transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),transforms.Normalize((53.858493788339445, 74.45490653753374,52.69974403185258), (63.94482059609943, 66.67201758796561,63.77821338039949))])
'''
data_transform = {
    "train": transforms.Compose([transforms.RandomResizedCrop(224),  #随机裁剪
                                 transforms.RandomHorizontalFlip(),  #随机翻转
                                 transforms.ToTensor(),             #转成tensor
                                 transforms.Normalize((53.77790797002807, 74.44627253046987, 52.712536131231), (63.931016292866104, 66.68932461497964, 63.8231957326705))]),   #all 0.5
    "val": transforms.Compose([transforms.Resize((224, 224)),
                               transforms.ToTensor(),
                               transforms.Normalize((54.58664421808152, 74.53292095850385, 52.584158276754714), (64.06482166566772, 66.51538124769947, 63.37020093276347))])}
'''
data_root = os.path.abspath(os.path.join(os.getcwd(), "../.."))  # get data root path
#image_path = data_root + "/data_set/flower_data/"  # flower data set path
#train_data = datasets.ImageFolder(root=image_path+"train", transform=data_transform["train"])
#train_data = datasets.ImageFolder(root=image_path+"flower_photos",transform = transform)
#train_loader = torch.utils.data.DataLoader(train_data, batch_size = batch_size, shuffle = True)
#print(train_data.train_data.size())
image_path = data_root +"/data_set/flower_data/flower_photos"
train_data = datasets.ImageFolder(root = image_path, transform = train_transform)
#train_loader = torch.utils.data.DataLoader(train_data, batch_size = batch_size, shuffle = True, num_workers = 0)

train_num = len(train_data)
# {'daisy':0, 'dandelion':1, 'roses':2, 'sunflower':3, 'tulips':4}
#flower_list = train_dataset.class_to_idx
# print(flower_list)
#%%
#cla_dict = dict((val, key) for key, val in flower_list.items())
# print(cla_dict)

#%%
# write dict into json file
#json_str = json.dumps(cla_dict, indent=4)
# print(json_str)
#with open('class_indices.json', 'w') as json_file:
#    json_file.write(json_str)

#%%
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0)    #num_workers线程个数，windows无法设置成非零值

#%%
#validate_dataset = datasets.ImageFolder(root=image_path + "val", transform=data_transform["val"])
#val_num = len(validate_dataset)
#validate_loader = torch.utils.data.DataLoader(validate_dataset,
#                                              batch_size=batch_size, shuffle=False,
#                                              num_workers=0)

# test_data_iter = iter(validate_loader)
# test_image, test_label = test_data_iter.next()

model_name = "vgg16"
net = vgg(model_name=model_name, num_classes=128, init_weights=True)
net.to(device)
loss_func = nn.MSELoss()
#loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.0001)

best_acc = 0.0
save_path = './{}Net.pth'.format(model_name)

for epoch in range(30):
   # net.train()
    running_loss = 0.0
    for step,data in enumerate(train_loader, start = 0):
        #b_x = Variable(x)
        images, labels = data
#        print(labels.shape)
#        print(labels[0])
#        print('\n')
        optimizer.zero_grad()
        middle, outputs = net(images.to(device))
        loss = loss_func(outputs,images.to(device))
        loss.backward()
        optimizer.step()
        #print('Epoch:',epoch,'| train loss: %.4f' % loss.item())

    # train
#    net.train()
#    for step, data in enumerate(train_loader, start=0):
#        images, labels = data
#        optimizer.zero_grad()
#        middle, outputs = net(images.to(device))
#        loss = loss_function(outputs, labels.to(device))
#        loss.backward()
#        optimizer.step()

        # print statistics
        running_loss += loss.item()
        # print train process
        rate = (step + 1) / len(train_loader)
        a = "*" * int(rate * 50)
        b = "." * int((1 - rate) * 50)
        print("\rtrain loss: {:^3.0f}%[{}->{}]{:.3f}".format(int(rate * 100), a, b, loss), end="")
    print('Epoch:',epoch,'| train loss: %.4f' % loss.item())
    print()
#putputs
np.set_printoptions(suppress=True)
with torch.no_grad():
    for step,data in enumerate(train_loader, start = 0):
        images, labels = data
        result, outputs = net(images.to(device)) 
        result = result.detach().cpu().numpy()
        a = np.array([0])
        a = torch.from_numpy(a)
        b = np.array([1])
        b = torch.from_numpy(b)
        c = np.array([2])
        c = torch.from_numpy(c)
        d = np.array([3])
        d = torch.from_numpy(d)
        for i in range(18):
            if labels[i]==a:
                filename = 'cgan.txt' 
            elif labels[i]==b:
                filename = 'gan.txt'
            elif labels[i]==c:
                filename = 'origin.txt'
            elif labels[i]==d:
                filename = 'tradition.txt'
            with open(filename,'a') as file_handle:
              
                for j in range(128):
                    file_handle.write(str(result[i][j]))
                    file_handle.write(' ')
                file_handle.write('\n')
        print(step,'/',len(train_loader),'\n')
'''
image_path = data_root +"/data_set/flower_data/flower_photos/"
gan_data = datasets.ImageFolder(root = image_path + 'gan', transform = train_transform)
gan_loader = torch.utils.data.DataLoader(gan_data,batch_size = batch_size,shuffle = False, num_workers = 0)
cgan_data = datasets.ImageFolder(root = image_path + 'cgan', transform = train_transform)
cgan_loader = torch.utils.data.DataLoader(cgan_data,batch_size = batch_size,shuffle = False, num_workers = 0)
origin_data = datasets.ImageFolder(root = image_path + 'origin', transform = train_transform)
origin_loader = torch.utils.data.DataLoader(origin_data,batch_size = batch_size,shuffle = False,num_workers =0)
tradition_data = datasets.ImageFolder(root = image_path +'tradition', transform = train_transform) 
tradition_loader = torch.utils.data.DataLoader(tradition_data, batch_size = batch_size, shuffle =False, num_workers = 0)

with torch.no_grad():
    for step,data in enumerate(gan_loader, start = 0):
        images, labels = data
        result, outputs = net(images.to(device))
        result = result.detach().numpy()
        with open('gan.txt','a') as file_handle:
            for i in range(30):
                for j in range(128):
                    file_handle.write(str(result[i][j]))
                    file_handle.write(' ')
                file_handle.write('\n')
        print('[',step,len(gan_loader),']','\n')

    for step,data in enumerate(cgan_loader, start = 0):
        images, labels = data
        result, outputs = net(images.to(device))
        result = result.detach().numpy()
        with open('cgan.txt','a') as file_handle:
            for i in range(30):
                for j in range(128):
                    file_handle.write(str(result[i][j]))
                    file_handle.write(' ')
                file_handle.write('\n')

    for step,data in enumerate(origin_loader):
        images, labels = data
        result, outputs = net(images.to(device))
        result = result.detach().numpy()
        with open('origin.txt','a') as file_handle:
            for i in range(30):
                for j in range(128):
                    file_handle.write(str(result[i][j]))
                    file_handle.write(' ')
                file_handle.write('\n')

    for step,data in enumerate(tradition_loader):
        images, labels =data
        result, outputs = net(images.to(device)) 
        result = result.detach().numpy()
        with open('tradition.txt','a') as file_handle:
            for i in range(30):
                for j in range(128):
                    file_handle.write(str(result[i][j]))
                    file_handle.write(' ')
                file_handle.write('\n')            
        print('[',step,len(tradition_loader),']','\n')
'''
'''
    # validate
    net.eval()
    acc = 0.0  # accumulate accurate number / epoch
    with torch.no_grad():
        for val_data in validate_loader:
            val_images, val_labels = val_data
            optimizer.zero_grad()
            outputs = net(val_images.to(device))
            predict_y = torch.max(outputs, dim=1)[1]
            acc += (predict_y == val_labels.to(device)).sum().item()
        val_accurate = acc / val_num
        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), save_path)
        print('[epoch %d] train_loss: %.3f  test_accuracy: %.3f' %
              (epoch + 1, running_loss / step, val_accurate))

print('Finished Training')

predict_transform = transforms.Compose(
    [transforms.Resize(224,224),
     transforms.ToTensor(),
     transforms.Normalize((53.858493788339445, 74.45490653753374, 52.69974403185258), (63.94482059609943, 66.67201758796561,63.77821338039949))])
'''
