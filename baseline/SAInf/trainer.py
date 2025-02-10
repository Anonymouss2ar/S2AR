import json
import logging
import os
import pickle

import numpy as np
import torch

from stay_area.baseline.SAInf.dataset import StayAreaDataset, collate_fn, collate_fn_test
from stay_area.baseline.SAInf.stayNet import train_epoch, StayNet, NoamOpt, loss_cal,  test_epoch
def test_model(device):
    with open("./data/test/test_data.pkl","rb") as f:
        test_data = pickle.load(f)
        f.close()
    # test_input_data=test_data
    with open("./data/road_network/node2gps.json","r") as f:
        node2gps=json.load(f)
        f.close()
    sad_test = StayAreaDataset(test_data)
    test_it = torch.utils.data.DataLoader(sad_test, collate_fn=collate_fn_test, batch_size=300)
    logging.info("finish data load")
    model=torch.load("./data/model/best_model.pt")
    model=model.to(device)
    logging.info("test")
    test_epoch(test_it, model, device,node2gps)

def train_model():
    with open("./data/train/train_data.pkl", "rb") as f:
        read_data = pickle.load(f)
        f.close()
    sad = StayAreaDataset(read_data)
    train_it = torch.utils.data.DataLoader(sad, collate_fn=collate_fn, batch_size=256)

    with open("./data/test/test_data.pkl",
              "rb") as f:
        test_data = pickle.load(f)
        f.close()
    with open("./data/road_network/node2gps.json","r") as f:
        node2gps=json.load(f)
        f.close()
    sad_test = StayAreaDataset(test_data)
    test_it = torch.utils.data.DataLoader(sad_test, collate_fn=collate_fn_test, batch_size=256)
    logging.info("finish data load")
    total_epoch=20
    model=StayNet(128,4,2,0.1)
    model_opt = NoamOpt(128, 0.5, 200,
                        # β1=0.9，β2=0.98
                        torch.optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.98), eps=1e-9))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for each_epoch in range(total_epoch):
        logging.info("epoch:{}".format(each_epoch))
        logging.info("train")
        loss_list=train_epoch(train_it,model,loss_cal(model_opt),device)
        torch.save(model, "./data/model/best_model.pt")
        logging.info(loss_list)
        logging.info("test")
        test_epoch(test_it,model,device,node2gps)
    torch.save(model, "./data/model/best_model.pt")
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format="[%(filename)s:%(lineno)s %(funcName)s()] -> %(message)s",
                        handlers=[
                            logging.FileHandler("./log/train_log.log",
                                                mode='w'),
                            logging.StreamHandler()]
                        )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_model()
