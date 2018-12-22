import sys
sys.path.append("..")

import dataloader.omniglot as omn_loader
from models.att import att_model
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler 
from tqdm import tqdm


def main(opt=None):
    data = omn_loader.load(["train"],opt)
    model = att_model(opt)
    model.train()
    if(opt['data.cuda']):
        model.cuda()
        
    optimizer = optim.Adam(model.parameters(),lr=0.0001)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)
    best_acc = 0
    file_object = open('acc_log.txt', 'w')


    for epoch in range(opt['train.epoch_num']):
        scheduler.step()
        for idx,image in enumerate(tqdm(data["train"])):
            loss,acc = model.forward(image)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if(acc>best_acc):
                best_acc = acc
                file_object.write(acc + '\n')
                
    file_object.close()
    
       
    torch.save(model.state_dict(),opt['model.save_path'] + '1.pth')
       