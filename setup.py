import torch
import time
import copy
import torch.nn as nn
import torchvision.models

def model():
    """
    :return:
        model base on ResNet50, change in fc layer
    """
    money_model = torchvision.models.resnet50(pretrained=True)
    in_features = money_model.fc.in_features
    money_model.fc = nn.Sequential(
        nn.Linear(in_features, 1000),
        nn.ReLU(inplace=True),
        nn.Dropout2d(0.5),
        nn.Linear(1000, 500),
        nn.ReLU(inplace=True),
        nn.Dropout2d(0.3),
        nn.Linear(500, 5)
    )
    return money_model

def train_model(dataloader, model, criterion, optimizer, n_epochs, dataset_size):
    """
    :param dataloader:
    :param model:
    :param criterion:
    :param optimizer:
    :param n_epochs:
    :return:
        model with best accurency
    """
    # set device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    since = time.time()
    iter_start = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(n_epochs+1):
        print('Epoch: {}/{}'.format(epoch+1, n_epochs))
        print('--'*20)
        iteration = 1
        # move network to device
        model.to(device)

        torch.backends.cudnn.benchmark = True
        # torch.backends.cudnn.deterministic = Flase

        for phase in ['train', 'val']:
            if phase == 'train':
                print('Training')
                model.train()
            else:
                print('--'*20)
                print('Evaluating')
                model.eval()

            running_loss = 0.0
            running_correct = 0.0

            for inputs, targets in dataloader[phase]:
                inputs = inputs.to(device)
                targets = targets.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, dim=1)
                    loss = criterion(outputs, targets)

                    if phase == 'train':
                        loss.backward() #tinh gradient cua cac weight
                        optimizer.step() #update cac weight the gradient decent

                    if iteration % 10 == 0:
                        iter_end = time.time()
                        duration = iter_end - iter_start
                        print('Iteration: {} || loss: {:.4f} || Duration 10 iter: {:.0f}m {:.0f}s'
                              .format(iteration, loss, duration // 60, duration % 60))
                        iter_start = time.time()

                running_loss += loss.item()
                running_correct += torch.sum(preds == targets.data)
                iteration += 1

            epoch_loss = running_loss / dataset_size[phase]
            epoch_acc = running_correct / dataset_size[phase]

            print('Loss in Epoch {}/{} is {:.4f}'.format(epoch+1, n_epochs, epoch_loss))
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    print('Traning complete in: {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best accurency: ', best_acc)

    torch.save(best_model_wts, 'money_weight.pth')
    model.load_state_dict(best_model_wts)
    return model

def predict(model, img):
    """
    :param model:
    :param img:
    :return:
        max accurency in classes prediction
        predict_class: int --> classes is encoded
    """
    output = model(img)
    accurency, predict_class = torch.max(output, dim=1)
    return accurency, predict_class




