import torch
import torch.nn as nn
import torch.nn.functional as F


from torch.autograd import Variable

from .utils import euclidean_dist

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)

class baseline_model(nn.Module):
    def __init__(self, opt):
        super(baseline_model,self).__init__()
        self.encoder = encoder(opt)
    
    def forward(self,sample):
        xs = Variable(sample['xs']) # support
        xq = Variable(sample['xq']) # query
        
        n_class = xs.size(0)
        n_support = xs.size(1)
        n_query = xq.size(1)
        
        target_inds = torch.arange(0, n_class).view(n_class, 1, 1).expand(n_class, n_query, 1).long()
        target_inds = Variable(target_inds, requires_grad=False)
        
        if xq.is_cuda:
            target_inds = target_inds.cuda()
            
        x = torch.cat([xs.view(n_class * n_support, *xs.size()[2:]),
                       xq.view(n_class * n_query, *xq.size()[2:])], 0)

        output = self.encoder.forward(x)
        
        output_dim = output.size(-1)
        
        z_proto = output[:n_class*n_support].view(n_class, n_support, output_dim).mean(1)
        zq = output[n_class*n_support:]

        dists = euclidean_dist(zq, z_proto)
        
        log_p_y = F.log_softmax(-dists, dim=1).view(n_class, n_query, -1)

        loss_val = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()

        _, y_hat = log_p_y.max(2)
        acc_val = torch.eq(y_hat, target_inds.squeeze()).float().mean()

        return loss_val, {
            'loss': loss_val.item(),
            'acc': acc_val.item()
        }
    
class encoder(nn.Module):
    def __init__(self,opt):
        super(encoder,self).__init__()
        self.model = nn.Sequential(
                self.make_conv_block(opt['model.input_dim'], opt['model.hid_dim']),
                self.make_conv_block(opt['model.hid_dim'], opt['model.hid_dim']),
                self.make_conv_block(opt['model.hid_dim'], opt['model.hid_dim']),
                self.make_conv_block(opt['model.hid_dim'], opt['model.out_dim']),
                Flatten()
                )
        
    
    def make_conv_block(self,in_channels,out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
    
    def forward(self,x):
        return self.model.forward(x)
        
    
    
        

   
    
        
        
        