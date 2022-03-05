import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from datasets import MyDataset, ImageTransform, TargetsTransform, make_datapath_list
from setup import*
import argparse

def main(args):
    # create model
    money_model = model()

    # pretrain weight
    if args.use_weights:
        print('Using weight pretrain to init model')
        print('--' * 20)
        money_weight = torch.load(args.weights)
        money_model.load_state_dict(money_weight)

    # setup loss_fn and optimization_fn
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(money_model.parameters(), lr=args.lr, momentum=0.9)

    path_list = make_datapath_list()
    label_path_list = []
    for label_path in path_list:
        label_path_list.append(label_path.split('/')[-2])
    train_path_idx, val_path_idx = train_test_split(list(range(len(label_path_list))), test_size=0.2, stratify=label_path_list)

    train_path = []
    val_path = []
    for i in train_path_idx:
        train_path.append(path_list[i])
    for j in val_path_idx:
        val_path.append(path_list[j])

    # creat dataLoader
    size = 224
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    train_dataset = MyDataset(train_path, transform=ImageTransform(size, mean, std), phase='train')
    val_dataset = MyDataset(val_path, transform=ImageTransform(size, mean, std), phase='val')

    trainLoader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.n_workers)
    valLoader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.n_workers)

    dataLoader = {'train': trainLoader, 'val': valLoader}
    dataset_sizes = {'train': len(train_dataset)//args.batch_size,
                     'val': len(val_dataset)//args.batch_size}

    # train_model
    train_model(dataLoader, money_model, criterion, optimizer, args.n_epochs, dataset_sizes)
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train money model')
    parser.add_argument('--use-weights', action='store_true', help='check use weights')
    parser.add_argument('--weights', type=str, default='money_weight.pth')
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--n-epochs', type=int, default=50)
    parser.add_argument('--n-workers', type=int, default=1)
    args = parser.parse_args()
    main(args)