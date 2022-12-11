from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from torchvision.models import resnet18, ResNet18_Weights
import matplotlib.pyplot as plt
import time
import os
import copy

# set up
cudnn.benchmark = True
plt.ion()   # interactive mode

"""
Returns dataloaders dictionary and class names list
"""
def load_data(data_dir='ACdata_base'):

    data_transforms = {
        'train': transforms.Compose([
            # transforms.Resize(512),
            transforms.CenterCrop(500),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]),
        'val': transforms.Compose([
            # transforms.Resize(512),
            transforms.CenterCrop(500),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]),
        'test': transforms.Compose([
            # transforms.Resize(512),
            transforms.CenterCrop(500),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    }

    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                            data_transforms[x])
                    for x in ['train', 'val', 'test']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                                shuffle=True, num_workers=4)
                for x in ['train', 'val', 'test']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'test']}
    class_names = image_datasets['train'].classes

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    return (dataloaders, class_names, device, dataset_sizes)


def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    # print("inp: ", inp)
    if title is not None:
        plt.title(title)
    plt.pause(3)  # pause a bit so that plots are updated



def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def fine_tune():
    # model_ft = models.resnet18(pretrained=True)
    model_ft = models.resnet18(weights=ResNet18_Weights.DEFAULT)

    # model_ft = models.vit_b_16(pretrained = True)

    # model_ft = models.alexnet(pretrained = True)

    num_ftrs = model_ft.fc.in_features
    # Here the size of each output sample is set to 2.
    # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
    model_ft.fc = nn.Linear(num_ftrs, len(class_names))

    model_ft = model_ft.to(device)

    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    return (model_ft, criterion, optimizer_ft, exp_lr_scheduler)


def visualize_model(model, num_images=10):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            # print("labels: ", labels)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title(f'predicted: {class_names[preds[j]]}' + f' label: {class_names[labels[j]]}') # DOUBLE CHECK HOW YOU ARE DOING LABELING
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)


def check_accuracy(model):
    was_training = model.training
    model.eval()

    num_correct = 0
    num_samples = 0
    num_classes = len(class_names)
    confusion_matrix = torch.zeros(num_classes, num_classes)

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['test']):

            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            num_correct += (preds == labels).sum()
            num_samples += preds.size(0)

            for j, (l, p) in enumerate(zip(labels, preds)):
                confusion_matrix[l.long(), p.long()] += 1
                # if class_names[l] == "naskh":
                #     ax = plt.subplot(1, 1, 1)
                #     ax.axis('off')
                #     ax.set_title(f'predicted: {class_names[p]}' + f' label: {class_names[l]}') # DOUBLE CHECK HOW YOU ARE DOING LABELING
                #     imshow(inputs.cpu().data[j])

        print(f'Got {num_correct} / {num_samples} with total accuracy {float(num_correct)/float(num_samples)*100:.2f}') 
        for k, class_accuracy in enumerate(list(confusion_matrix.diag() / confusion_matrix.sum(1))):
            print(f'{class_names[k]} accuracy: {class_accuracy}')
            
        model.train(mode=was_training)


if __name__ == "__main__":
    
    training_mode = False

    
    dataloaders, class_names, device, dataset_sizes = load_data()
    model_ft, criterion, optimizer_ft, exp_lr_scheduler = fine_tune()
    # Get a batch of training data
    # inputs, classes = next(iter(dataloaders['train']))
    PATH = "./lstmmodelgpu_2.pt"

    if training_mode:
        print("start training.................")
        
        # Make a grid from batch
        # out = torchvision.utils.make_grid(inputs)
        # imshow(out, title=[class_names[x] for x in classes])

        model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                        num_epochs=20)  
        
        torch.save(model_ft.state_dict(), PATH)
    else:
        model_ft.load_state_dict(torch.load(PATH))
        model_ft.eval()

    # visualize_model(model_ft)
    # print("Classes: ", class_names[5])
    check_accuracy(model_ft)
