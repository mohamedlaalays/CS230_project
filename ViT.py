
import argparse
from pathlib import Path

import timm
import timm.data
import timm.loss
import timm.optim
import timm.utils
import torch
import torchmetrics
from timm.scheduler import CosineLRScheduler
from pytorch_accelerated import Trainer
import torch.nn as nn
from torch.optim import lr_scheduler
from pathlib import Path

# from pytorch_accelerated.callbacks import SaveBestModelCallback
# from pytorch_accelerated.trainer import Trainer, DEFAULT_CALLBACKS


def create_datasets(image_size, data_mean, data_std, train_path, val_path):
    train_transforms = timm.data.create_transform(
        input_size=image_size,
        is_training=True,
        mean=data_mean,
        std=data_std,
        auto_augment="rand-m7-mstd0.5-inc1",
    )

    eval_transforms = timm.data.create_transform(
        input_size=image_size, 
        mean=data_mean, 
        std=data_std
    )

    train_dataset = timm.data.dataset.ImageDataset(
        train_path, transform=train_transforms
    )
    eval_dataset = timm.data.dataset.ImageDataset(val_path, transform=eval_transforms)

    return train_dataset, eval_dataset

def get_test_dataset(image_size, data_mean, data_std, path):

    test_transforms = timm.data.create_transform(
        input_size=image_size, 
        mean=data_mean, 
        std=data_std
    )

    test_dataset = timm.data.dataset.ImageDataset(path, transform=test_transforms)

    return test_dataset

def main():

    training_mode = True
    
    data_path = Path("./dataset")
    train_path = data_path / "train"
    val_path = data_path / "val"
    test_path = data_path / "test"
    image_size = (224, 224)
    num_classes = len(list(train_path.iterdir())) # NUM OF CLASSES IS OFF BY ONE COS OF THE .DS_Store
    print("Num classes: ", num_classes)
    print("Classes: ", list(train_path.iterdir()))

    model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=num_classes)

    # print("Last model layer: ", model_ft.get_classifier())
    # print("Model: ", model_ft.default_cfg)

    data_config = timm.data.resolve_data_config({}, model=model, verbose=True)
    data_mean = data_config["mean"]
    data_std = data_config["std"]
 

    lr = 5e-3
    momentum = 0.8

    train_dataset, eval_dataset = create_datasets(
        train_path=train_path,
        val_path=val_path,
        image_size=image_size,
        data_mean=data_mean,
        data_std=data_std,
    )

    # print("data_config: ", data_config)
    # print("train_dataset: ", len(train_dataset))

    loss_func = nn.CrossEntropyLoss()
    optimizer = timm.optim.create_optimizer_v2(model, opt='sgd', lr=lr, momentum=momentum)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1) # YOU SHOULD CONSIDER USING THIS!!!

    trainer = Trainer(
        model,
        loss_func=loss_func,
        optimizer=optimizer,
        )

    save_path = "./lstmmodelgpu_2.pt"

    if training_mode:
        trainer.train(
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            num_epochs=2,
            per_device_batch_size=32,
        )
        trainer.save_checkpoint(save_path)
    else:
        trainer.load_checkpoint(save_path)
        model.eval()
    test_dataset = get_test_dataset(image_size, data_mean, data_std, test_path)
    Trainer.evaluate(test_dataset)

    


if __name__ == "__main__":
    main()