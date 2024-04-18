import models
import dataset
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.io import read_image
from torch.optim import Adam
from torch.utils.data import DataLoader, Subset
from torchsummary import summary
import os
import json

def run():

    STUDY_NAME = "vgg11_cifar10_dual_late"

    model = "STCNN"
    DUAL_INPUT = True
    merge_at = "late"

    RUN_TRAIN = False
    RUN_TEST = True
    STORE_MODEL = False

    # Dataset
    number_of_data_batches = 5 # Unit: 10000 training images

    transform = transforms.Compose([
            transforms.ToTensor(), # Convert the PIL image to tensor
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    if DUAL_INPUT:
        cifar10_dataset = dataset.CIFAR10Dataset(dataset_path="dataset/cifar-10", transform=transform, batches=number_of_data_batches, dual_input=True)
        train_data = Subset(cifar10_dataset, range(0, cifar10_dataset.get_train_size()))
        test_data = Subset(cifar10_dataset, range(cifar10_dataset.get_train_size(), len(cifar10_dataset)))
    else:
        cifar10_dataset = dataset.CIFAR10Dataset(dataset_path="dataset/cifar-10", transform=transform, batches=number_of_data_batches)
        train_data = Subset(cifar10_dataset, range(0, cifar10_dataset.get_train_size()))
        test_data = Subset(cifar10_dataset, range(cifar10_dataset.get_train_size(), len(cifar10_dataset)))

    # Settings
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if DUAL_INPUT:
        if model == "STCNN":
            model = models.DualSTCNN(input_size=cifar10_dataset[0][0].shape, num_classes=len(cifar10_dataset.classes), merge_at=merge_at)
        elif model == "VGG11":
            model = models.DualVGG11(input_size=cifar10_dataset[0][0].shape, num_classes=len(cifar10_dataset.classes), merge_at=merge_at)
        elif model == "VGG16":
            model = models.DualVGG16(input_size=cifar10_dataset[0][0].shape, num_classes=len(cifar10_dataset.classes), merge_at=merge_at)
        else:
            raise ValueError("Model not found")
    else:
        if model == "STCNN":
            model = models.STCNN(input_size=cifar10_dataset[0][0].shape, num_classes=len(cifar10_dataset.classes))
        elif model == "VGG11":
            model = models.VGG11(input_size=cifar10_dataset[0][0].shape, num_classes=len(cifar10_dataset.classes))
        elif model == "VGG16":
            model = models.VGG16(input_size=cifar10_dataset[0][0].shape, num_classes=len(cifar10_dataset.classes))
        else:
            raise ValueError("Model not found")
        
    loss_fn = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=0.001)

    # Training
    batch_size = 32
    epochs = 10

    if RUN_TRAIN:
        run_train(model, loss_fn, optimizer, train_data, DUAL_INPUT, device, batch_size, epochs)

    # Test run
    RUN_SINGLE_TEST = True
    index_if_single_test = 1

    if RUN_TEST:
        test_accuracy = 0
        test_loss = 0
        if RUN_SINGLE_TEST:
            test_accuracy, test_loss = run_test(model, loss_fn, test_data, cifar10_dataset, DUAL_INPUT, device, is_batch=False, index=index_if_single_test)
        else:     
            test_accuracy, test_loss = run_test(model, loss_fn, test_data, cifar10_dataset, DUAL_INPUT, device, is_batch=True)

    if STORE_MODEL:
        if not os.path.exists(f"./results/{STUDY_NAME}"):
            os.makedirs(f"./results/{STUDY_NAME}")
        torch.save(model.state_dict(), f"./results/{STUDY_NAME}/model.pth")
        record = {
            "study_name": STUDY_NAME,
            "model": model.__class__.__name__,
            "dual_input": DUAL_INPUT,
            "merge_at": merge_at,
            "dataset": cifar10_dataset.__class__.__name__,
            "data_size": len(cifar10_dataset),
            "test_accuracy": test_accuracy if RUN_TEST else None,
            "test_loss": test_loss if RUN_TEST else None,

            "loss_fn": loss_fn.__class__.__name__,
            "optimizer": optimizer.__class__.__name__ if RUN_TRAIN else None,
            "batch_size": batch_size if RUN_TRAIN else None,
            "epochs": epochs if RUN_TRAIN else None,
        }
        with open(f"./results/{STUDY_NAME}/record.json", "w") as f:
            json.dump(record, f)


def run_test(model, loss_fn, test_data, cifar10_dataset, DUAL_INPUT, device, is_batch=False, index=0):
    if is_batch:
        test_loader = DataLoader(test_data, batch_size=32, shuffle=True)
        model.to(device)
        model.eval()
    else:
        test_data = [test_data[index]]
        test_loader = DataLoader(test_data, batch_size=1, shuffle=True)
        model.to(device)
        model.eval()

    correct = 0
    total = 0
    running_loss = 0.0

    with torch.no_grad():
        if DUAL_INPUT:
            for images, images2, labels in test_loader:
                images = images.to(device)
                images2 = images2.to(device)
                labels = labels.to(device)
                output = model(images, images2)
                loss = loss_fn(output, labels)
                correct += (torch.argmax(output, dim=1) == labels).sum().item()
                running_loss += loss.item()
                total += images.shape[0]
        else:
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)
                output = model(images)
                loss = loss_fn(output, labels)
                correct += (torch.argmax(output, dim=1) == labels).sum().item()
                running_loss += loss.item()
                total += images.shape[0]
            
        if not is_batch:
            print(f"Predicted class: {cifar10_dataset.get_class(torch.argmax(output, dim=1).item())}")
            print(f"True class: {cifar10_dataset.get_class(labels.item())}")
            cifar10_dataset.show_image(index)

    print(f"Accuracy: {correct / total * 100:.4f}%")
    print(f"Loss: {running_loss / total:.4f}")

    return correct / total, running_loss / total

def run_train(model, loss_fn, optimizer, train_data, DUAL_INPUT, device, batch_size, epochs):
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    model.to(device)
    model.train()

    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        for i, data in enumerate(train_loader):
            if DUAL_INPUT:
                images, images2, labels = data
                images = images.to(device)
                images2 = images2.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                output = model(images, images2)
                loss = loss_fn(output, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                correct += (torch.argmax(output, dim=1) == labels).sum().item()
                total += images.shape[0]
            else:
                images, labels = data
                images = images.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                output = model(images)
                loss = loss_fn(output, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                correct += (torch.argmax(output, dim=1) == labels).sum().item()
                total += images.shape[0]

            if i % 100 == 99:
                print(f"Epoch {epoch+1}, batch {i+1}, loss: {running_loss / total}, accuracy: {correct / total}")
                running_loss = 0.0
                correct = 0
                total = 0

        print(f"Epoch {epoch+1} finished: loss: {running_loss / total:.4f}, accuracy: {correct / total*100:.4f}%")

    print("Finished training")

if __name__ == "__main__":
    run()