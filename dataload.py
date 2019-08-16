import cv2
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.autograd import Variable

class KittiDataset(Dataset):
    def __init__(self, train_file, mode='train', n_samples=None):
        self.mode = mode
        self.train_file = open(train_file,'r')
        self.image_lists = []
        for line in self.train_file:
            line = line.strip('\n')
            self.image_lists.append(line)
        self.data_root = '/home/zzg/Datasets/kitti'

    def __len__(self):
        return len(self.image_lists)

    def __getitem__(self, i):
        image_name = self.image_lists[i]
        #image_name = '000002_10'
        
        left_path = self.data_root + '/image_2/' + image_name + '.png'
        left_image = cv2.imread(left_path,0)
        
        disp_path = self.data_root + '/disp_occ_0/' + image_name + '.png'
        disp_image = cv2.imread(disp_path,0)
        
        cost_path = self.data_root + '/leftbin/' + image_name + '.bin'
        
        max_pixel = np.max(left_image)
        disp_val,x,y =0,0,0 
        img_h , img_w = left_image.shape

        left_image = left_image.astype(np.float32)
        
        while(disp_val==0 or disp_val>69):
            x = random.randint(400,img_w-300)
            y = random.randint(125, img_h-125)
            disp_val = disp_image[y,x]

        image_patchs_l = []
        patch_cods_l = []
        
        image_patchs_r = []
        patch_cods_r = []
        
        image_patchs_u = []
        patch_cods_u = []
        
        image_patchs_d = []
        patch_cods_d = []
        
        for i in range(120,x):#left
            x0 = i-3 
            x1 = i+4
            y0 = y-3
            y1 = y+4
            img_patch = left_image[y0:y1, x0:x1]
            patch_mean = np.mean(img_patch)
            img_patch_normlz = (img_patch - patch_mean)/max_pixel
            
            image_patchs_l.append(img_patch_normlz)
            patch_cods_l.append([i,y])

        for i in range(x+1,img_w-4):#right
            x0 = i-3 
            x1 = i+4
            y0 = y-3
            y1 = y+4
            img_patch = left_image[y0:y1, x0:x1]
            patch_mean = np.mean(img_patch)
            img_patch_normlz = (img_patch - patch_mean)/max_pixel
            
            image_patchs_r.append(img_patch_normlz)
            patch_cods_r.append([i,y])
            
        for j in range(3,y):#up
            x0 = x-3
            x1 = x+4
            y0 = j-3
            y1 = j+4
            img_patch = left_image[y0:y1, x0:x1]
            patch_mean = np.mean(img_patch)
            img_patch_normlz = (img_patch - patch_mean)/max_pixel
            
            image_patchs_u.append(img_patch_normlz)
            patch_cods_u.append([x,j])
            
        for j in range(y+1,img_h-4):#down
            x0 = x-3
            x1 = x+4
            y0 = j-3
            y1 = j+4
            img_patch = left_image[y0:y1, x0:x1]
            patch_mean = np.mean(img_patch)
            img_patch_normlz = (img_patch - patch_mean)/max_pixel
            
            image_patchs_d.append(img_patch_normlz)
            patch_cods_d.append([x,j])

        origin_patch = []
        origin_cod = []
        x0 = x-3
        x1 = x+4
        y0 = y-3
        y1 = y+4
        img_patch = left_image[y0:y1, x0:x1]
        patch_mean = np.mean(img_patch)
        img_patch_normlz = (img_patch - patch_mean)/max_pixel
        origin_patch.append(img_patch_normlz)
        origin_cod.append([x,y])

        ### left path
        image_patchs_l = np.array(image_patchs_l)
        image_patchs_l = torch.tensor(image_patchs_l)
        image_patchs_l = image_patchs_l.unsqueeze(1)  ## [n,1,5,5]
        image_patchs_l = image_patchs_l.type(torch.FloatTensor)

        patch_cods_l = np.array(patch_cods_l)
        patch_cods_l = torch.tensor(patch_cods_l)
        patch_cods_l = patch_cods_l.type(torch.FloatTensor)
        patch_cods_l[:,0]=patch_cods_l[:,0]/img_w
        patch_cods_l[:,1]=patch_cods_l[:,1]/img_h

        ### right path
        image_patchs_r = np.array(image_patchs_r)
        image_patchs_r = torch.tensor(image_patchs_r)
        image_patchs_r = image_patchs_r.unsqueeze(1)  ## [n,1,5,5]
        image_patchs_r = image_patchs_r.type(torch.FloatTensor)

        patch_cods_r = np.array(patch_cods_r)
        patch_cods_r = torch.tensor(patch_cods_r)
        patch_cods_r = patch_cods_r.type(torch.FloatTensor)
        patch_cods_r[:,0]=patch_cods_r[:,0]/img_w
        patch_cods_r[:,1]=patch_cods_r[:,1]/img_h
        
        ### up path
        image_patchs_u = np.array(image_patchs_u)
        image_patchs_u = torch.tensor(image_patchs_u)
        image_patchs_u = image_patchs_u.unsqueeze(1)  ## [n,1,5,5]
        image_patchs_u = image_patchs_u.type(torch.FloatTensor)

        patch_cods_u = np.array(patch_cods_u)
        patch_cods_u = torch.tensor(patch_cods_u)
        patch_cods_u = patch_cods_u.type(torch.FloatTensor)
        patch_cods_u[:,0]=patch_cods_u[:,0]/img_w
        patch_cods_u[:,1]=patch_cods_u[:,1]/img_h
        
        ### down path
        image_patchs_d = np.array(image_patchs_d)
        image_patchs_d = torch.tensor(image_patchs_d)
        image_patchs_d = image_patchs_d.unsqueeze(1)  ## [n,1,5,5]
        image_patchs_d = image_patchs_d.type(torch.FloatTensor)

        patch_cods_d = np.array(patch_cods_d)
        patch_cods_d = torch.tensor(patch_cods_d)
        patch_cods_d = patch_cods_d.type(torch.FloatTensor)
        patch_cods_d[:,0]=patch_cods_d[:,0]/img_w
        patch_cods_d[:,1]=patch_cods_d[:,1]/img_h

        ### x0,y0 
        origin_patch = np.array(origin_patch)
        origin_patch = torch.tensor(origin_patch)
        origin_patch = origin_patch.unsqueeze(1)  ## [n,1,5,5]
        origin_patch = origin_patch.type(torch.FloatTensor)
        
        origin_cod = np.array(origin_cod)
        origin_cod = torch.tensor(origin_cod)
        origin_cod = origin_cod.type(torch.FloatTensor)
        origin_cod[:,0]=origin_cod[:,0]/img_w
        origin_cod[:,1]=origin_cod[:,1]/img_h

        ### match cost
        d_cost = np.memmap(cost_path, dtype=np.float32, shape=(1, 70, img_h, img_w))
        p = np.isnan(d_cost)
        d_cost[p] = 0.0  
        d_cost = torch.tensor(d_cost)
        d_cost = d_cost.squeeze(0)
        d_cost = d_cost.type(torch.FloatTensor)  # [70,h,w]

#        d_cost_l = d_cost[:,y,120:x]
#        d_cost_r = d_cost[:,y,x+1:img_w-3]
#        d_cost_u = d_cost[:,2:y,x]
#        d_cost_d = d_cost[:,y+1:img_h-3,x]
        d_cost_l = d_cost[:,y,120:x+1]
        d_cost_r = d_cost[:,y,x:img_w-4]
        d_cost_u = d_cost[:,3:y+1,x]
        d_cost_d = d_cost[:,y:img_h-4,x]
        d_cost_x0 = d_cost[:,y,x]

        return image_patchs_l, patch_cods_l, image_patchs_r, patch_cods_r, image_patchs_u, patch_cods_u, image_patchs_d, patch_cods_d, origin_patch, origin_cod ,disp_val, d_cost_l , d_cost_r, d_cost_u, d_cost_d,d_cost_x0 #, x, y

