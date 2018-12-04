import sys
sys.path.append("..")
import dataloader.omniglot as omn_loader


def main(opt=None):
    data = omn_loader.load(["train"])
    return data

if __name__=="__main__":
    print(main())