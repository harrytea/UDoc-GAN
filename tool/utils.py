import numpy as np
import torch
import random
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from collections import OrderedDict
from PIL import Image
import os
import torch.distributed as dist
from torch.nn import init
from torchvision import transforms



'''  initial seed  '''
def init_seeds(seed=0):
    random.seed(seed)  # seed for module random
    np.random.seed(seed)  # seed for numpy
    torch.manual_seed(seed)  # seed for PyTorch CPU
    torch.cuda.manual_seed(seed)  # seed for current PyTorch GPU
    torch.cuda.manual_seed_all(seed)  # seed for all PyTorch GPUs
    if seed == 0:
        # if True, causes cuDNN to only use deterministic convolution algorithms.
        torch.backends.cudnn.deterministic = True
        # if True, causes cuDNN to benchmark multiple convolution algorithms and select the fastest.
        torch.backends.cudnn.benchmark = False



'''  distributed mode  '''
def initial_distributed():
    '''  initial distributed mode  '''
    if torch.cuda.is_available() is False:
        raise EnvironmentError("not find GPU device for training")
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        gpu = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        print("os.environ[\"WORLD_SIZE\"]: ", os.environ["WORLD_SIZE"])
        print("os.environ[\"RANK\"]: ", os.environ["RANK"])
        print("os.environ[\"LOCAL_RANK\"]: ", os.environ["LOCAL_RANK"])
    else:
        print("Not using distributed mode")
    torch.cuda.set_device(rank)
    dist_url = 'env://'
    dis_backend = 'nccl'  # communication: nvidia GPU recommened nccl
    print('| distributed init (rank {}): {}'.format(rank, dist_url), flush=True)
    dist.init_process_group(backend=dis_backend, init_method=dist_url, world_size=world_size, rank=rank)
    dist.barrier()
    return rank



'''convert tensor to numpy and concat them
    - input:  B, C, H, W (tensor*N)
    - output: H*N, W, C  numpy
    - return: PIL Image
'''
def tensor2cnumpy(*images):
    res = []
    for img in images:
        img  = img.detach().cpu().numpy()[0]*0.5+0.5
        res.append(img)
    result = np.concatenate(res, axis=2)
    result = np.transpose(result, (1, 2, 0))
    if result.shape[2]==1:  # gray images
        result = np.concatenate([result, result, result], axis=2)
        result = Image.fromarray(np.clip(result * 255.0, 0, 255.0).astype('uint8'))
        return result
    result = Image.fromarray(np.clip(result * 255.0, 0, 255.0).astype('uint8'))
    return result

def tensor2cnumpy_ndetach(*images):
    res = []
    for img in images:
        img  = img[0]*0.5+0.5
        res.append(img)
    result = np.concatenate(res, axis=2)
    result = np.transpose(result, (1, 2, 0))
    result = Image.fromarray(np.clip(result * 255.0, 0, 255.0).astype('uint8'))
    return result

'''convert tensor to numpy
    - input:  B, C, H, W (tensor)
    - output: H, W, C  numpy
'''
def tensor2numpy(image):
    image  = image.detach().cpu().numpy()[0]*0.5+0.5
    image  = np.transpose(image, (1, 2, 0))
    image = np.clip(image * 255.0, 0, 255.0).astype('uint8')
    return image


def fix_model_state_dict(state_dict):
    """
    remove 'module.' of dataparallel
    """
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k
        if name.startswith("module."):
            name = name[7:]
        new_state_dict[name] = v
    return new_state_dict


def set_requires_grad(nets, requires_grad=False):
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad



def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.
    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.
    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>



class ReplayBuffer():
    def __init__(self, max_size=50):
        assert (max_size > 0), 'Empty buffer or trying to create a black hole. Be careful.'
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        to_return = []
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if random.uniform(0,1) > 0.5:
                    i = random.randint(0, self.max_size-1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return torch.cat(to_return)



class QueueBGC():
    def __init__(self, length):
        self.max_length = length
        self.queue = []

    def insert(self, mask):
        if self.queue.__len__() >= self.max_length:
            self.queue.pop(0)

        self.queue.append(mask)

    def rand_item(self):
        assert self.queue.__len__() > 0, 'Error! Empty queue!'
        return self.queue[np.random.randint(0, self.queue.__len__())]

    def last_item(self):
        assert self.queue.__len__() > 0, 'Error! Empty queue!'
        return self.queue[self.queue.__len__()-1]


class SaveValues():
    def __init__(self, layer):
        self.model  = None
        self.input  = None
        self.output = None
        self.forward_hook  = layer.register_forward_hook(self.hook_fn_act)
    def hook_fn_act(self, module, input, output):
        self.model  = module
        self.output = output
    def remove(self):
        self.forward_hook.remove()
