from pytorch_grad_cam import GradCAM, HiResCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from twoandahalfdimensions.utils.config import  Config
import numpy as np
from matplotlib import pyplot as plt
import cv2
import torch

def cam(model, image):
    b_sz, C, N_z, N_x, N_y = image.shape #The batch contains only one image in this case, it can also contain multiple images
    
    for name, param in model.named_parameters():
        param.requires_grad = True
    
    target_layers = [model.model[1].layer4[-1]]

    #cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True)
    cam = HiResCAM(model=model, target_layers=target_layers, use_cuda=True)

    grayscale_cam = cam(input_tensor=image) #ahpe(b_sz,H,W,D)

    print('shape of gray', grayscale_cam.shape)

    grayscale_cam = grayscale_cam[0, :] #first element of the batch 

    if C != 3:
        rgb_img = np.zeros((image.shape[2], image.shape[3],image.shape[4],3))
        rgb_img[:,:,:,0] = image[0,0,:,:,:]
        rgb_img[:,:,:,1] = image[0,0,:,:,:]
        rgb_img[:,:,:,2] = image[0,0,:,:,:]

    visualization = show_cam_on_image(img=rgb_img, mask=grayscale_cam, use_rgb=True) #shape (224,224,128,3)
    #print('visualization', visualization.shape)
    

    #cam = image[0,0,:,:,64] #one slice of the image 
    #cam = cam / np.max(cam)
    #cv2.imwrite('image_slice.png', np.uint8(255 * cam))

    cv2.imwrite('heatmap_slice.png', visualization[:,:,64,:])

