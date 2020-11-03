import torch
import argparse
import unet
from torch import optim
from dataset import MyDataset, TestDataset
from torch.utils.data import DataLoader
import numpy as np
import cv2


def train(model):
    model.train()
    batch_size = args.batch_size
    # loss_fn = torch.nn.BCELoss()  # 二分类
    loss_fn = torch.nn.CrossEntropyLoss()  # 多分类
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    train_dataset = MyDataset("data/train/imgs", "data/train/masks")

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    for epoch in range(args.num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, args.num_epochs))
        print('-' * 10)
        dataset_size = len(train_dataloader.dataset)
        epoch_loss = 0
        step = 0
        for x, y, _, __ in train_dataloader:
            inputs = x.to(device)
            labels = y.to(device)
            outputs = model(inputs)
            loss = loss_fn(outputs, torch.squeeze(labels, 1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            step += 1
            print("%d/%d,train_loss:%0.5f" % (step, dataset_size // train_dataloader.batch_size, loss.item()))
        print("epoch %d loss:%0.5f" % (epoch, epoch_loss))
        torch.save(model.state_dict(), 'epoch/epoch_%d.pth' % epoch)


def test(model):
    model.eval()
    test_dataset = TestDataset("data/test/imgs")
    test_dataloader = DataLoader(test_dataset)

    with torch.no_grad():
        for x, x_path in test_dataloader:
            y = model(x.to(device))
            img_y = torch.squeeze(y)
            img_y = torch.max(img_y, 0)[0]
            img_y = img_y.ceil()

            print(x_path)
            img_y[img_y == 0] = 0
            img_y[img_y == 1] = 42
            img_y[img_y == 2] = 85
            img_y[img_y == 3] = 127
            img_y[img_y == 4] = 170
            img_y[img_y == 5] = 255
            img_y = img_y.cpu().to(torch.uint8)
            img = np.asarray(img_y)
            cv2.imwrite(x_path[0][0:-4] + '_pred.png', img)


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = unet.UNet(1, 6).to(device)
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', dest='type', type=str, default='train', help='train or test')
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=1, help='batch_size')
    parser.add_argument('--load', dest='load', type=str, help='the path of the .pth file')
    parser.add_argument('--epoch', dest='num_epochs', type=int, default=20, help='num_epochs')
    parser.add_argument('--lr', dest='learning_rate', type=float, default=1e-3, help='learning_rate')
    args = parser.parse_args()
    if args.load:
        model.load_state_dict(torch.load(args.load, map_location=device))
        print('加载预训练模型:{}'.format(args.load))
    if args.type == 'train':
        train(model)
    elif args.type == 'test':
        test(model)
