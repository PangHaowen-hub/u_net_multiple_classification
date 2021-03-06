import torch
import argparse
import vnet
import nnvnet
from torch import optim
from dataset import MyDataset
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast as autocast

def train(model):
    model.train()
    # model.load_state_dict(torch.load(args.load, map_location='cuda'))
    batch_size = args.batch_size
    loss_fn = torch.nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    train_dataset = MyDataset("data/train/imgs")
    # val_percent = 0.3
    # n_val = int(len(train_dataset) * val_percent)
    # n_train = len(train_dataset) - n_val
    # train, val = torch.utils.data.random_split(train_dataset, [n_train, n_val])
    # train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    # val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True, drop_last=True)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    for epoch in range(args.num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, args.num_epochs))
        print('-' * 10)
        dataset_size = len(train_dataloader.dataset)
        epoch_loss = 0
        step = 0
        for x, y in train_dataloader:
            inputs = x.to(device)
            labels = y.to(device)
            optimizer.zero_grad()
            with autocast():
                output = model(inputs)
                loss = loss_fn(output, labels)
                loss.backward()
                optimizer.step()
            epoch_loss += loss.item()
            step += 1
            print(_)
            print(__)
            print("%d/%d,train_loss:%0.5f" % (step, dataset_size // train_dataloader.batch_size, loss.item()))
        print("epoch %d loss:%0.5f" % (epoch, epoch_loss))
        torch.save(model.state_dict(), 'epoch_%d.pth' % epoch)


# def test(model):
#     model.eval()
#     model.load_state_dict(torch.load(args.load, map_location='cuda'))
#     test_dataset = TestDataset("data/test/imgs")
#     test_dataloader = DataLoader(test_dataset)
#     palette = [[0], [42], [85], [127], [170], [255]]
#     with torch.no_grad():
#         for x, x_path in test_dataloader:
#             y = model(x.to(device))
#             img_y = torch.squeeze(y).to('cpu').numpy() > 0.5
#             img_y = img_y.transpose([1, 2, 0])
#             img_y = onehot_to_mask(img_y, palette)
#             print(x_path)
#             cv2.imwrite(x_path[0][0:-4] + '_pred.png', img_y)


# def onehot_to_mask(mask, palette):
#     x = np.argmax(mask, axis=-1)
#     colour_codes = np.array(palette)
#     x = np.uint8(colour_codes[x.astype(np.uint8)])
#     return x


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')

    model = nnvnet.VNet(nll=True).to(device)
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', dest='type', type=str, default='train', help='train or test')
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=1, help='batch_size')
    parser.add_argument('--load', dest='load', type=str, help='the path of the .pth file')
    parser.add_argument('--epoch', dest='num_epochs', type=int, default=100, help='num_epochs')
    parser.add_argument('--lr', dest='learning_rate', type=float, default=1e-3, help='learning_rate')
    args = parser.parse_args()

    if args.type == 'train':
        train(model)
    # elif args.type == 'test':
    #     test(model)
