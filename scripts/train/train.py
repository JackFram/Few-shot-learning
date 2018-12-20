import sys
sys.path.append("..")
import dataloader.omniglot as omn_loader
from models.baseline import baseline_model,encoder
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler 


def main(opt=None):
    data = omn_loader.load(["train"],opt)
    model = baseline_model(encoder(1,64,64))
    model.train()
    if(opt['data.cuda']):
        model.cuda()
    optimizer = optim.Adam(model.parameters(),lr=0.0001)
    for idx,image in enumerate(data["train"]):
        loss,acc = model.forward(image)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(loss,acc)
        if(idx==10):
            break
        

if __name__=="__main__":
    main()