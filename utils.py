import time
import copy
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from torchvision import datasets, models, transforms
from sklearn.metrics import classification_report
import seaborn as sn
import pandas as pd
from sklearn.metrics import confusion_matrix


"""
Returns dataloaders dictionary and class names list
"""
def load_data(data_dir, data_transforms):

    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                            data_transforms[x])
                    for x in ['train', 'val', 'test']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                                shuffle=True, num_workers=4) # NUMBER OF WORKERS PREVIOUSLY WAS 4
                for x in ['train', 'val', 'test']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'test']}
    class_names = image_datasets['train'].classes
    num_classes = len(class_names)

    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    return (dataloaders, num_classes, class_names, dataset_sizes)




def train_model(model, device, dataloaders, dataset_sizes, criterion, optimizer, scheduler, num_epochs):
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



def check_accuracy(model, device, dataloaders, class_names, architecture):
    was_training = model.training
    model.eval()

    num_correct = 0
    num_samples = 0
    num_classes = len(class_names)
    my_confusion_matrix = torch.zeros(num_classes, num_classes)

    with torch.no_grad():
        pred_y, true_y = [], []
        for i, (inputs, labels) in enumerate(dataloaders['test']):

            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            pred_y.extend(preds.data.cpu().numpy())
            true_y.extend(labels.data.cpu().numpy())

            num_correct += (preds == labels).sum()
            num_samples += preds.size(0)

            for j, (l, p) in enumerate(zip(labels, preds)):
                my_confusion_matrix[l.long(), p.long()] += 1
                # if class_names[l] == "naskh":
                #     ax = plt.subplot(1, 1, 1)
                #     ax.axis('off')
                #     ax.set_title(f'predicted: {class_names[p]}' + f' label: {class_names[l]}') # DOUBLE CHECK HOW YOU ARE DOING LABELING
                #     imshow(inputs.cpu().data[j])

        print(f'Got {num_correct} / {num_samples} with total accuracy {float(num_correct)/float(num_samples)*100:.2f}') 
        for k, class_accuracy in enumerate(list(my_confusion_matrix.diag() / my_confusion_matrix.sum(1))):
            print(f'{class_names[k]} accuracy: {class_accuracy}')
            
        print("Confusion matrix: ")
        print(my_confusion_matrix)

        print("Classification report:")
        cl_rep = classification_report(true_y, pred_y, digits=3)
        print(cl_rep)
        # df_cl_rep = pd.DataFrame(cl_rep)
        # plt.figure(figsize = (12,7))
        # sn.heatmap(df_cl_rep, annot=True)
        # plt.savefig("./images/"+ "class_" + architecture + '.png')

        cf_matrix = confusion_matrix(true_y, pred_y)
        df_cm = pd.DataFrame(cf_matrix/np.sum(cf_matrix) *9, index = [i for i in class_names],
                            columns = [i for i in class_names])

        plt.figure(figsize = (12,7))
        sn.heatmap(df_cm, annot=True)
        plt.savefig("./images/" + "conf_" + architecture + '.png')

        model.train(mode=was_training)



def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.5, 0.5, 0.5])
    std = np.array([0.5, 0.5, 0.5])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    # print("inp: ", inp)
    if title is not None:
        plt.title(title)
    plt.pause(3)  # pause a bit so that plots are updated


def visualize_model_pred(model, device, dataloaders, class_names, num_images=10):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)

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