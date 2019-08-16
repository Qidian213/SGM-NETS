import cv2
import random
import numpy as np
import torch
from readpfm import readPFM

class KittiDataset():
    def __init__(self, train_file, mode='train', n_samples=None):
        self.mode = mode

        self.left_image = cv2.imread('/home/wsy/datasets/training/image_2/000004_10.png',0)
        self.left_image = self.left_image.astype(np.float32)
        self.max_pixel = np.max(self.left_image)
       
        self.disp_image = cv2.imread('/home/wsy/datasets/training/disp_occ_0/000004_10.png',0)
        self.disp_image = self.disp_image.astype(np.int32)
        
        self.ct_image = cv2.imread('/home/wsy/datasets/training/13*13ct/disp/4disp_ct.png',0)
        self.ct_image = self.ct_image.astype(np.int32)
       
        self.d_cost = np.load('/home/wsy/datasets/training/13*13ct/dcost/4.npy')
        self.d_cost = np.array(self.d_cost)
        self.d_cost = torch.tensor(self.d_cost)
        self.d_cost = self.d_cost.type(torch.FloatTensor)

        self.img_h , self.img_w = self.left_image.shape
        self.vaid_points = []
       
        for i in range(320, self.img_w-200):
            for j in range(250, self.img_h - 50):
                disp_val = self.disp_image[j,i]
                if(disp_val>3 and disp_val < 70):
                    self.vaid_points.append([i,j,disp_val])
                   
        random.shuffle(self.vaid_points)
        self.vaid_points = random.sample(self.vaid_points, int(len(self.vaid_points)/5))
        
        print(self.d_cost.size())
        print(self.img_h,self.img_w)

    def getlen(self):
        return len(self.vaid_points)

    def getitem(self, i):
        
        train_point = self.vaid_points[i]
        x = train_point[0]
        y = train_point[1]
        disp_val = train_point[2]

        image_patchs_l = []
        patch_cods_l = []
        
        image_patchs_r = []
        patch_cods_r = []
        
        image_patchs_u = []
        patch_cods_u = []
        
        image_patchs_d = []
        patch_cods_d = []
        
        spare_idex = []
        
        xl = 0
        yl = 0
        xr = 0
        yr = 0
        xu = 0
        yu = 0
        xd = 0
        yd = 0

        #left
        for i in range(x-50,120,-1):
        
            dispgt = self.disp_image[y,i]
            dispct = self.ct_image[y,i]
            if dispgt!=0:
                spare_idex.append(i)
            if dispct!=0 and dispgt!=0 and abs(dispgt-dispct)<1:
                xl = i
                yl = y
                break               
        if xl == 0:
            if len(spare_idex) !=0:                
                xl = spare_idex[len(spare_idex)-1]
                yl = y
            else:
                yl = y
                xl = x-200
        spare_idex.clear()
        #right
        for i in range(x+50,self.img_w-3):
        
            dispgt = self.disp_image[y,i]
            dispct = self.ct_image[y,i]
            if dispgt!=0:
                spare_idex.append(i)
            if dispct!=0 and dispgt!=0 and abs(dispgt-dispct)<1:
                xr = i
                yr = y
                break               
        if xr == 0:
            if len(spare_idex) !=0:                
                xr = spare_idex[len(spare_idex)-1]
                yr = y
            else:
                yr = y
                xr = x+(self.img_w-x-4)
        spare_idex.clear()        
        
        #up
        for j in range(y-30,3,-1):
        
            dispgt = self.disp_image[j,x]
            dispct = self.ct_image[j,x]
            if dispgt!=0:
                spare_idex.append(j)
            if dispct!=0 and dispgt!=0 and abs(dispgt-dispct)<1:
                xu = x
                yu = j
                break               
        if yu == 0:
            if len(spare_idex) !=0:                
                yu = spare_idex[len(spare_idex)-1]
                xu = x
            else:
                yu = y-100
                xu = x
        spare_idex.clear() 
        #down
        for j in range(y+30,self.img_h-3):
        
            dispgt = self.disp_image[j,x]
            dispct = self.ct_image[j,x]
            if dispgt!=0:
                spare_idex.append(j)
            if dispct!=0 and dispgt!=0 and abs(dispgt-dispct)<1:
                xd = x
                yd = j
                break               
        if yd == 0:
            if len(spare_idex) !=0:                
                yd = spare_idex[len(spare_idex)-1]
                xd = x
            else:
                yd = y+(self.img_h-y-4)
                xd = x
        spare_idex.clear() 
     
     
     
        print(xl,yl)
        print(xr,yr)
        print(xu,yu)
        print(xd,yd)
        for i in range(xl,x):#left
            x0 = i-3 
            x1 = i+4
            y0 = y-3
            y1 = y+4
            img_patch = self.left_image[y0:y1, x0:x1]
            patch_mean = np.mean(img_patch)
            img_patch_normlz = (img_patch - patch_mean)/self.max_pixel
            
            image_patchs_l.append(img_patch_normlz)
            patch_cods_l.append([i,y])

        for i in range(x+1,xr):#right
            x0 = i-3 
            x1 = i+4
            y0 = y-3
            y1 = y+4
            img_patch = self.left_image[y0:y1, x0:x1]
            patch_mean = np.mean(img_patch)
            img_patch_normlz = (img_patch - patch_mean)/self.max_pixel
            
            image_patchs_r.append(img_patch_normlz)
            patch_cods_r.append([i,y])
            
        for j in range(yu,y):#up
            x0 = x-3
            x1 = x+4
            y0 = j-3
            y1 = j+4
            img_patch = self.left_image[y0:y1, x0:x1]
            patch_mean = np.mean(img_patch)
            img_patch_normlz = (img_patch - patch_mean)/self.max_pixel
            
            image_patchs_u.append(img_patch_normlz)
            patch_cods_u.append([x,j])
            
        for j in range(y+1,yd):#down
            x0 = x-3
            x1 = x+4
            y0 = j-3
            y1 = j+4
            img_patch = self.left_image[y0:y1, x0:x1]
            patch_mean = np.mean(img_patch)
            img_patch_normlz = (img_patch - patch_mean)/self.max_pixel
            
            image_patchs_d.append(img_patch_normlz)
            patch_cods_d.append([x,j])

        origin_patch = []
        origin_cod = []
        x0 = x-3
        x1 = x+4
        y0 = y-3
        y1 = y+4
        img_patch = self.left_image[y0:y1, x0:x1]
        patch_mean = np.mean(img_patch)
        img_patch_normlz = (img_patch - patch_mean)/self.max_pixel
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
        patch_cods_l[:,0]=patch_cods_l[:,0]/self.img_w
        patch_cods_l[:,1]=patch_cods_l[:,1]/self.img_h

        ### right path
        image_patchs_r = np.array(image_patchs_r)
        image_patchs_r = torch.tensor(image_patchs_r)
        image_patchs_r = image_patchs_r.unsqueeze(1)  ## [n,1,5,5]
        image_patchs_r = image_patchs_r.type(torch.FloatTensor)

        patch_cods_r = np.array(patch_cods_r)
        patch_cods_r = torch.tensor(patch_cods_r)
        patch_cods_r = patch_cods_r.type(torch.FloatTensor)
        patch_cods_r[:,0]=patch_cods_r[:,0]/self.img_w
        patch_cods_r[:,1]=patch_cods_r[:,1]/self.img_h
        
        ### up path
        image_patchs_u = np.array(image_patchs_u)
        image_patchs_u = torch.tensor(image_patchs_u)
        image_patchs_u = image_patchs_u.unsqueeze(1)  ## [n,1,5,5]
        image_patchs_u = image_patchs_u.type(torch.FloatTensor)

        patch_cods_u = np.array(patch_cods_u)
        patch_cods_u = torch.tensor(patch_cods_u)
        patch_cods_u = patch_cods_u.type(torch.FloatTensor)
        patch_cods_u[:,0]=patch_cods_u[:,0]/self.img_w
        patch_cods_u[:,1]=patch_cods_u[:,1]/self.img_h
        
        ### down path
        image_patchs_d = np.array(image_patchs_d)
        image_patchs_d = torch.tensor(image_patchs_d)
        image_patchs_d = image_patchs_d.unsqueeze(1)  ## [n,1,5,5]
        image_patchs_d = image_patchs_d.type(torch.FloatTensor)

        patch_cods_d = np.array(patch_cods_d)
        patch_cods_d = torch.tensor(patch_cods_d)
        patch_cods_d = patch_cods_d.type(torch.FloatTensor)
        patch_cods_d[:,0]=patch_cods_d[:,0]/self.img_w
        patch_cods_d[:,1]=patch_cods_d[:,1]/self.img_h

        ### x0,y0 
        origin_patch = np.array(origin_patch)
        origin_patch = torch.tensor(origin_patch)
        origin_patch = origin_patch.unsqueeze(1)  ## [n,1,5,5]
        origin_patch = origin_patch.type(torch.FloatTensor)
        
        origin_cod = np.array(origin_cod)
        origin_cod = torch.tensor(origin_cod)
        origin_cod = origin_cod.type(torch.FloatTensor)
        origin_cod[:,0]=origin_cod[:,0]/self.img_w
        origin_cod[:,1]=origin_cod[:,1]/self.img_h

        ### match cost
        d_cost_l = self.d_cost[:,y,xl:x+1]
        d_cost_r = self.d_cost[:,y,x:xr]
        d_cost_u = self.d_cost[:,yu:y+1,x]
        d_cost_d = self.d_cost[:,y:yd,x]
        d_cost_x0 = self.d_cost[:,y,x]

        return image_patchs_l, patch_cods_l, image_patchs_r, patch_cods_r, image_patchs_u, patch_cods_u, image_patchs_d, patch_cods_d, origin_patch, origin_cod ,disp_val, d_cost_l , d_cost_r, d_cost_u, d_cost_d,d_cost_x0, x, y

