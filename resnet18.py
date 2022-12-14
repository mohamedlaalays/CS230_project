from __future__ import print_function, division
from platform import architecture
import timm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
from utils import load_data, train_model, visualize_model_pred, check_accuracy


def ViT_data_transforms():
   
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]),
        'val': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]),
        'test': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    }

    return data_transforms


def fine_tune(num_classes, device):

    model_ft = timm.create_model('resnet18', pretrained=True, num_classes=num_classes)

    model_ft = model_ft.to(device)

    return model_ft



if __name__ == "__main__":
    
    training_mode = True

    # set up
    cudnn.benchmark = True
    # plt.ion()   # interactive mode

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    data_transforms = ViT_data_transforms()

    data_dir = "ACdata_base_original"
    # data_dir = "ACdata_base"
    # data_dir = "ACdata_base_no_train"


    dataloaders, num_classes, class_names, dataset_sizes = load_data(data_dir, data_transforms)

    PATH = "./resnet18_epoch10_original.pt"     # ==> Original AC dataset
    # PATH = "./resnet18_epoch10_compl.pt"      # ==> AC dataset + Calliar dataset evenly distributed in train, eval, test
    # PATH = "./resnet18_epoch10_no_train.pt"   # ==> AC dataset + Calliar dataset only in eval and test
    
    architecture = "resnet18_epoch10_original" 
    # architecture = "resnet18_epoch10_compl"
    # architecture = "resnet18_epoch10_no_train"

    model_ft = fine_tune(num_classes, device)

    # print(model_ft.default_cfg)

    if training_mode:
        
        print("start training.................")

        criterion = nn.CrossEntropyLoss()
        optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

        model_ft = train_model(model_ft, device, dataloaders,
                            dataset_sizes, criterion, 
                            optimizer_ft, exp_lr_scheduler,
                            num_epochs=10
                            )
        torch.save(model_ft.state_dict(), "./trained_models/"+PATH)  
    else:
        model_ft.load_state_dict(torch.load("./trained_models/"+PATH))
        model_ft.eval()

    # visualize_model_pred(model_ft, device, dataloaders, class_names, num_images=10)
    check_accuracy(model_ft, device, dataloaders, class_names, architecture)
