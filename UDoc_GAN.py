import os

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

import torch
import logging
import argparse
from tqdm import tqdm
from tool.utils import *
from itertools import chain
from tool.logger import get_root_logger
from tool.dataset import CustomDataset
from tool.model import LPNet, Generator3, Generator6, Discriminator



parser = argparse.ArgumentParser(description='')
'''  train  '''
parser.add_argument('--batch_size',     default=16,     type=int,    help='number of samples in one batch')
parser.add_argument('--beta1',          default=0.5,    type=float,  help='momentum term of adam')
parser.add_argument('--lr',             default=0.0002, type=float,  help='initial learning rate for adam')
parser.add_argument('--epoch_count',    default=1,      type=int,    help='the starting epoch count')
parser.add_argument('--n_epochs',       default=100,    type=int,    help='number of epochs with the initial learning rate')
parser.add_argument('--n_epochs_decay', default=500,    type=int,    help='number of epochs to linearly decay learning rate to zero')
'''  param  '''
parser.add_argument('--input_nc',       default=6,      type=int,    help='input image channels: 3-Document 3-background')
parser.add_argument('--output_nc',      default=3,      type=int,    help='output image channels: 3-Document')
parser.add_argument('--crop_size',      default=256,    type=int,    help='then crop to this size')
parser.add_argument('--model',          default="UDoc-GAN",          help='decise which data to choose')
'''  about save  '''
parser.add_argument('--data_dir',       default='/data4/wangyh/doc/wyh/dataset')
parser.add_argument('--ckpt_dir',       default='./ckpts/udoc',help='directory for checkpoints')
parser.add_argument('--local_rank',     default=0,                   help='if use distributed mode, must use variable local_rank')
'''  about loss  '''
parser.add_argument('--lambda_A',       default=10.0,   type=float,  help='weight for cycle loss (A -> B -> A)')
parser.add_argument('--lambda_B',       default=10.0,   type=float,  help='weight for cycle loss (B -> A -> B)')
parser.add_argument('--lambda_identity',default=0.5,    type=float,  help='identity mapping.')
args = parser.parse_args()



def main():
    '''  1. initial distributed mode  '''
    rank = initial_distributed()
    logger = get_root_logger(name='GAN', log_file=os.path.join(args.ckpt_dir, "train.log"), log_level=logging.INFO)


    '''  2. logger  '''
    if not os.path.exists(args.ckpt_dir):
        os.makedirs(args.ckpt_dir)


    '''  3. datasets  '''
    train_dataset       = CustomDataset(args.data_dir, args.crop_size)
    train_sampler       = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_batch_sampler = torch.utils.data.BatchSampler(train_sampler, batch_size=args.batch_size, drop_last=True)
    train_loader        = torch.utils.data.DataLoader(dataset=train_dataset, batch_sampler=train_batch_sampler, num_workers=8, pin_memory=False)
    if rank==0:
        logger.info('Number of training data: {}'.format(len(train_dataset)))


    '''  4. initial model  '''
    lpnet        = LPNet(in_channels=3)
    state_dict   = torch.load("./ckpts/lpnet/best_LPNet.pth", map_location=torch.device('cpu'))
    lpnet.load_state_dict(fix_model_state_dict(state_dict))
    lpnet.cuda().eval()
    netG_A = Generator3(in_channels=3, out_channels=args.output_nc).cuda()
    netG_B = Generator6(in_channels=args.input_nc, out_channels=args.output_nc).cuda()
    netD_A = Discriminator().cuda()
    netD_B = Discriminator().cuda()
    init_weights(netG_A)
    init_weights(netG_B)
    init_weights(netD_A)
    init_weights(netD_B)
    netG_A = torch.nn.parallel.DistributedDataParallel(netG_A, device_ids=[rank], broadcast_buffers=False)
    netG_B = torch.nn.parallel.DistributedDataParallel(netG_B, device_ids=[rank], broadcast_buffers=False)
    netD_A = torch.nn.parallel.DistributedDataParallel(netD_A, device_ids=[rank], broadcast_buffers=False)
    netD_B = torch.nn.parallel.DistributedDataParallel(netD_B, device_ids=[rank], broadcast_buffers=False)
    if rank==0:
        logger.info("netG parameters: {}".format(sum(param.numel() for param in netG_A.parameters())/1e6))
        logger.info("netD parameters: {}".format(sum(param.numel() for param in netD_A.parameters())/1e6))


    '''  5. optimizer loss scheduler  '''
    # loss
    criterionGAN   = torch.nn.MSELoss()
    criterionCycle = torch.nn.L1Loss()
    criterionIdt   = torch.nn.L1Loss()
    # optimizer
    optimizer_G = torch.optim.Adam(chain(netG_A.parameters(), netG_B.parameters()), lr=args.lr,   betas=(args.beta1, 0.999))
    optimizer_D = torch.optim.Adam(chain(netD_A.parameters(), netD_B.parameters()), lr=2*args.lr, betas=(args.beta1, 0.999))
    def lambda_rule(epoch):
        lr_l = 1.0 - max(0, epoch + args.epoch_count - args.n_epochs) / float(args.n_epochs_decay + 1)
        return lr_l
    # scheduler
    scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=lambda_rule)
    scheduler_D = torch.optim.lr_scheduler.LambdaLR(optimizer_D, lr_lambda=lambda_rule)
    # Buffer and target
    target_real   = torch.ones((args.batch_size, 1, 30, 30), dtype=torch.float32, requires_grad=False).cuda()
    target_fake   = torch.zeros((args.batch_size, 1, 30, 30), dtype=torch.float32, requires_grad=False).cuda()
    fake_A_buffer = ReplayBuffer(max_size=50)
    fake_B_buffer = ReplayBuffer(max_size=50)
    bgc_queue     = QueueBGC(train_loader.__len__()/4)


    '''  6. train  '''
    for epoch in range(args.epoch_count, args.n_epochs+args.n_epochs_decay+1):
        train_sampler.set_epoch(epoch)

        '''  log learning rate  '''
        if rank==0:
            logger.info("epoch: {} lr_G: {} lr_D: {}".format(epoch, optimizer_G.param_groups[0]["lr"], optimizer_D.param_groups[0]["lr"]))

        loss_G_all = 0
        loss_D_all = 0
        loss_G_Id_all = 0
        loss_G_GAN_all = 0
        loss_G_Cycle_all = 0
        # A: abnormal ill image  &&  B: normal ill img
        for iter, (fileA, fileB, real_A, real_B) in enumerate(tqdm(train_loader), 0):
            real_A, real_B = real_A.cuda(), real_B.cuda()
            # predict background color
            with torch.no_grad():
                color = lpnet(real_A)
                back_color = torch.ones_like(real_A) * (color.unsqueeze(2).unsqueeze(3))

            ''' Generator '''
            set_requires_grad([netD_A, netD_B], False)
            optimizer_G.zero_grad()
            # Identity loss
            same_B = netG_A(real_B)     # B-GA-B
            loss_identity_B = criterionIdt(same_B, real_B) * 5.0
            same_A = netG_B(real_A, back_color)    # A-GB-A
            loss_identity_A = criterionIdt(same_A, real_A) * 5.0
            # GAN loss
            fake_B = netG_A(real_A)     # A-GA-B
            pred_fake_A  = netD_A(fake_B)
            loss_GAN_A2B = criterionGAN(pred_fake_A, target_real)
            bgc_queue.insert(back_color)           # B-GB-A
            fake_A = netG_B(real_B, bgc_queue.rand_item())
            pred_fake_B  = netD_B(fake_A)
            loss_GAN_B2A = criterionGAN(pred_fake_B, target_real)
            # Cycle loss
            recovered_A    = netG_B(fake_B, bgc_queue.last_item())
            loss_cycle_ABA = criterionCycle(recovered_A, real_A) * 10.0
            recovered_B    = netG_A(fake_A)
            loss_cycle_BAB = criterionCycle(recovered_B, real_B) * 10.0
            # Total loss
            loss_G = loss_identity_A + loss_identity_B + loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB
            loss_G.backward()
            optimizer_G.step()
            loss_G_all = loss_G_all+loss_G.item()
            loss_G_Id_all = loss_G_Id_all+loss_identity_A.item()+loss_identity_B.item()
            loss_G_GAN_all = loss_G_GAN_all+loss_GAN_A2B.item()+loss_GAN_B2A.item()
            loss_G_Cycle_all = loss_G_Cycle_all+loss_cycle_ABA.item()+loss_cycle_BAB.item()

            ''' Discriminator '''
            set_requires_grad([netD_A, netD_B], True)
            # loss D_B
            optimizer_D.zero_grad()
            pred_real = netD_B(real_A)
            loss_D_A_real = criterionGAN(pred_real, target_real) # real
            fake_A = fake_A_buffer.push_and_pop(fake_A)
            pred_fake = netD_B(fake_A.detach())
            loss_D_A_fake = criterionGAN(pred_fake, target_fake) # fake
            loss_D_A = (loss_D_A_real + loss_D_A_fake) * 0.5
            # loss D_A
            pred_real = netD_A(real_B)
            loss_D_B_real = criterionGAN(pred_real, target_real) # real
            fake_B = fake_B_buffer.push_and_pop(fake_B)
            pred_fake = netD_A(fake_B.detach())
            loss_D_B_fake = criterionGAN(pred_fake, target_fake) # fake
            loss_D_B = (loss_D_B_real + loss_D_B_fake) * 0.5
            loss_D = loss_D_A+loss_D_B
            loss_D.backward()
            optimizer_D.step()
            loss_D_all = loss_D_all+loss_D.item()

        if rank==0:
            logger.info("epoch:{} G_loss:{:.5f} D_loss:{:.5f}".format(epoch, loss_G_all, loss_D_all))

        '''   log model   '''
        if rank==0 and epoch%1==0:
            torch.save(netG_A.state_dict(), os.path.join(args.ckpt_dir, 'epoch{}_netG_A.pth'.format(epoch)))
            # torch.save(netG_B.state_dict(), os.path.join(args.ckpt_dir, 'epoch{}_netG_B.pth'.format(epoch)))
            # torch.save(netD_A.state_dict(), os.path.join(args.ckpt_dir, 'epoch{}_netD_A.pth'.format(epoch)))
            # torch.save(netD_B.state_dict(), os.path.join(args.ckpt_dir, 'epoch{}_netD_B.pth'.format(epoch)))

        scheduler_G.step()
        scheduler_D.step()


if __name__ == '__main__':
    init_seeds(5634)
    main()