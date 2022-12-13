from __future__ import print_function, division

import torch
import timm
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
from torchvision import transforms
from utils import load_data, train_model, visualize_model_pred, check_accuracy


def ViT_data_transforms():
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]),
        'val': transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]),
        'test': transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])
    }

    return data_transforms


def fine_tune(num_classes, device):

    model_ft = timm.create_model('inception_v3', pretrained=True, num_classes=num_classes)
    model_ft = model_ft.to(device)
    
    return model_ft



if __name__ == "__main__":
    
    training_mode = True

    # set up
    cudnn.benchmark = True
    # plt.ion()   # interactive mode

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    data_transforms = ViT_data_transforms()
    data_dir = "ACdata_base"
    dataloaders, num_classes, class_names, dataset_sizes = load_data(data_dir, data_transforms)
    PATH = "./inception_v3_epoch25.pt"

    model_ft = fine_tune(num_classes, device)

    print(model_ft.default_cfg)

    if training_mode:
        
        print("start training.................")

        criterion = nn.CrossEntropyLoss()
        optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

        model_ft = train_model(model_ft, device, dataloaders,
                            dataset_sizes, criterion, 
                            optimizer_ft, exp_lr_scheduler,
                            num_epochs=25
                            )
        torch.save(model_ft.state_dict(), PATH)  
    else:
        model_ft.load_state_dict(torch.load(PATH))
        model_ft.eval()

    # visualize_model_pred(model_ft, device, dataloaders, class_names, num_images=10)
    # print("Classes: ", class_names[5])
    check_accuracy(model_ft, device, dataloaders, class_names)
