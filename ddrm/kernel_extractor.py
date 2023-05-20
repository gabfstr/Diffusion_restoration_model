import os
import sys
import numpy as np

sys.path.append('../')
from NonUniformBlurKernelEstimation.models.TwoHeadsNetwork import TwoHeadsNetwork

import torch
from torchvision import transforms


def combine_kernels_threshold(kernels, masks, threshold=2.5, verboose=False):
    # kernels of size [1 K kh kw]
    # masks of size [1 K h w]
    _, h, w = masks[0].shape
    threshold_pct= threshold*h*w/100

    pix=masks[0].sum(dim=[1,2])
    args=np.argwhere(pix.cpu()>threshold_pct)
    #print("Sum of pixels in each kernel : ",pix.sort(descending=True)) if verboose else None
    print("{} indices found to be > threshold_pct : {}".format(args.shape[1],args[0])) if verboose else None
    one_kernel=kernels[0,args].sum(dim=1)
    #print("Final combinated kernel of shape : ",one_kernel.shape)
    return one_kernel


def low_rank_approx(m, rank = 1):
  U,E,V = torch.linalg.svd(m)
  mn = torch.zeros_like(m)
  score = 0.0
  for i in range(rank):
    mn += E[0,i] * torch.outer(U[0,:,i], V[0,i,:])
    score += E[0,i]
  print('Kernel approximation percentage :', (score / E.sum()).item())
  if rank==1:
    s=torch.sqrt(E[0,0])
    return mn, s*U[0,:,0],s*V[0,0,:]
  return mn, None, None


def get_blur_kernel(img, model_path, output_dir="testing_results", gpu_id=0, gamma_factor=2.2):

    # img is [1 c h w]

    K = 25 # number of elements in the base
    blurry_image = img[0].permute(1,2,0).cpu().numpy()
    model_file = model_path

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)


    # Expected : [w h 3] on cpu ?
    two_heads = TwoHeadsNetwork(K).cuda(gpu_id)
    two_heads.load_state_dict(torch.load(model_file, map_location='cuda:%d' % gpu_id))

    two_heads.eval()

    # Put output to tensor
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    blurry_tensor = transform(blurry_image).cuda(gpu_id)


    # Kernels and masks are estimated
    blurry_tensor_to_compute_kernels = blurry_tensor**gamma_factor - 0.5
    kernels_estimated, masks_estimated = two_heads(blurry_tensor_to_compute_kernels[None,:,:,:])

    return kernels_estimated, masks_estimated