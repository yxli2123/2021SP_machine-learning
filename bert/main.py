import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from dataloader import Reviews
from model import Transformer


def main(args):
    device = args.device
    train_set = Reviews(args.model, 'train')
    train_dataloader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=True)

    model = Transformer(args.model, args.cls_num)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        for t, batch in enumerate(train_dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}
            logit = model(batch)
            loss = criterion(logit, batch['cls'])
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
    torch.save(model.state_dict(), './ckpt.pth')


if __name__ == '__main__':
    args = {"model": "roberta-large",
            "cls_num": 5,
            'batch_size': 16,
            'lr': 1e-5,
            'device': 'cuda:0',
            'epochs': 100}
