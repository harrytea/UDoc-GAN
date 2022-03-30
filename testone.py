import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import argparse
from tool.model import Generator3
import torch
from tool.dataset import TestUDocNet
from tool.utils import *
from tqdm import tqdm


parser = argparse.ArgumentParser(description='')
parser.add_argument('--dataroot',       default='/data4/wangyh/doc/wyh',    help='true: takes images in order to make batches, otherwise takes them randomly')
parser.add_argument('--ckpt_dir',       default='./ckpts/',          help='directory for checkpoints')
args = parser.parse_args()

def main():
    '''  datasets  '''
    args.dataroot = "/data4/wangyh/doc/wyh"
    test_dataset  = TestUDocNet(args.dataroot)
    test_loader   = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=1, num_workers=8, pin_memory=False)
    print('Number of testing data: %d' % len(test_dataset))

    '''  model && loss '''
    netG_A = Generator3().cuda()
    netG_A_dict = torch.load("./ckpts/udoc/epoch190_netG_A.pth")
    netG_A.load_state_dict(fix_model_state_dict(netG_A_dict))

    '''  start testing!  (dont need model.eval())'''
    with torch.no_grad():
        for i, (file, image, padw, padh) in enumerate(tqdm(test_loader), 0):
            real_A = image.cuda()
            path_A = file

            fake_B = netG_A(real_A)
            '''   log images every 10 images  '''
            if i%1==0:
                real_A, fake_B = real_A[:,:,0:-padh,0:-padw], fake_B[:,:,0:-padh,0:-padw]
                fake_B = tensor2numpy(fake_B)
                fake_B = Image.fromarray(fake_B)
                fake_B.save("./epoch190/"+path_A[0])


if __name__ == '__main__':
    main()