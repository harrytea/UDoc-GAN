import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6"

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


    # eee = [111,112,113,114,115,116,117,118,119,120,121,122,123,124,125,126,127,128,129,130,131,132,133,134,135,136,137,138,139,140,141,142,143,144,145,146,147,148,149]
    '''  model && loss '''
    for ee in range(582, 583, 1):
        netG_A = Generator3().cuda()
        netG_A_dict = torch.load("./ckpts/udoc/epoch"+str(ee)+"_netG_A.pth")
        netG_A.load_state_dict(fix_model_state_dict(netG_A_dict))

        if not os.path.exists("./doctr/epoch{}/".format(ee)):
            os.makedirs("./doctr/epoch{}/".format(ee))

        '''  start testing!  (dont need model.eval())'''
        with torch.no_grad():
            for i, (file, image, padw, padh) in enumerate(tqdm(test_loader), 0):
                real_A = image.cuda()  # 1. real_A
                path_A = file

                fake_B = netG_A(real_A)
                '''   log images every 10 images  '''
                real_A, fake_B = real_A[:,:,0:-padh,0:-padw], fake_B[:,:,0:-padh,0:-padw]
                fake_B = tensor2numpy(fake_B)
                fake_B = Image.fromarray(fake_B)
                fake_B.save("./doctr/epoch{}/".format(ee)+path_A[0])


if __name__ == '__main__':
    main()