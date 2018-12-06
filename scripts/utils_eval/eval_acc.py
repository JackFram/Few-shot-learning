import torch
import torch.nn.functional as F

class evaluator:
    def __init__(self,true_label,log_softmax_p):
        '''
        arguments:
        true_label :  should be a matrix of size(n_class,n_query,1)
        log_softmax_p : should be a matrix of size(n_class,n_query,n_class)
        '''
        self.true_label = true_label
        self.log_softmax_p = log_softmax_p
        
    def acc(self):
        _, y_hat = self.log_softmax_p.max(2)
        acc_val = torch.eq(self.y_hat, self.true_label.squeeze()).float().mean()
        return acc_val
        
        