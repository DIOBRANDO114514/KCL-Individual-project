import torch
import torchvision
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from torch.nn.utils import prune
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.image as image
import cv2
import os
import time
from sklearn.metrics import classification_report



DATA_ROOT = '/content/ycb_rgb256'

# Initialisation of weights
def weight_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


# Define the training function
def train(epoch, network, train_loader, optimizer, device, train_losses, train_counter, train_acces, log_interval):
    network.train()  # Setting the network to training mode
    train_correct = 0
    # For a set of batch
    example_inputs = torch.randn(1, 3, 32, 32)
    example_inputs = example_inputs.to(device)

    # Ignore layers that do not need to be pruned, e.g. the final classification layer
    ignored_layers = []
    for m in network.modules():
        if isinstance(m, torch.nn.Linear):
            ignored_layers.append(m)  # DO NOT prune the final classifier!
    predicted_labels = []

    for batch_idx, (data, target) in enumerate(train_loader):
        # Get batch_id, data, and label by enumerate
        # 1-Zeroing the gradient
        optimizer.zero_grad()

        # 2-Pass in an image of a batch and forward compute it.
        # data.to(device) puts the image into the GPU for computation.
        output = network(data.to(device))
        # 3-Calculation of losses
        loss = F.nll_loss(output, target.to(device))
        # 4-Backward propagation
        loss.backward()
        # 5-Optimisation parameters
        optimizer.step()
        # exp_lr_scheduler.step()

        train_pred = output.data.max(dim=1, keepdim=True)[1]  # Take the largest category in output
        # dim = 1 means to go to the maximum value of each row, and [1] means to take the index of the maximum value without going to the maximum value itself [0].

        train_correct += train_pred.eq(target.data.view_as(train_pred).to(device)).sum()  # Compare and find the number of correct classifications.
        # Print the following information: epoch, number of images, total number of training images, percentage of completion, current loss.
        print('\r No. {} Train Epoch: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, (batch_idx + 1) * len(data), len(train_loader.dataset),
                   100. * batch_idx / len(train_loader), loss.item()), end='')

        # Every 10th batch (log_interval = 10)
        if batch_idx % log_interval == 0:
            # print(batch_idx)
            # Add the current loss to train_losses
            train_losses.append(loss.item())
            # Count
            train_counter.append(
                (batch_idx * data.size(0)) + ((epoch - 1) * len(train_loader.dataset)))
    train_acc = train_correct / len(train_loader.dataset)
    train_acces.append(train_acc.cpu().numpy().tolist())
    print('\tTrain Accuracy:{:.2f}%'.format(100. * train_acc))


# Defining Test Functions
def test(epoch, network, test_loader, device, test_acces, test_losses, optimizer, true_labels, predicted_labels):
    network.eval()  # Set the network to evaluating mode.
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = network(data.to(device))  # Pass in this set of batch for forward computation.
            true_labels.extend(target.tolist())
            predicted_labels.extend(output.argmax(dim=1).tolist())

            test_loss += F.nll_loss(output, target.to(device), reduction='sum').item()

            pred = output.data.max(dim=1, keepdim=True)[1]  # Take the largest category in output.
            # dim = 1 means to go to the maximum value of each row, and [1] means to take the index of the maximum value without going to the maximum value itself [0].

            correct += pred.eq(target.data.view_as(pred).to(device)).sum()  # Compare and find the number of correct classifications.
    acc = correct / len(test_loader.dataset)  # Average test accuracy.
    test_acces.append(acc.cpu().numpy().tolist())

    test_loss /= len(test_loader.dataset)  # The average loss, len, is 10000.
    test_losses.append(test_loss)  # Records the test_loss for the epoch.

    # Preserve the model with the greatest test accuracy.
    if test_acces[-1] >= max(test_acces):
        # Save the model after each batch is trained.
        torch.save(network.state_dict(), './model02.pth')

        # Save the optimiser after each batch is trained.
        torch.save(optimizer.state_dict(), './optimizer02.pth')

    print('\r Test set \033[1;31m{}\033[0m : Avg. loss: {:.4f}, Accuracy: {}/{}  \033[1;31m({:.2f}%)\033[0m\n' \
          .format(epoch, test_loss, correct, len(test_loader.dataset), 100. * acc), end='')

    return test_loss

# CNN MODEL
class BasicBlock(nn.Module):
    expansion = 1 # Expansion factor, used in calculating the number of output features

    def __init__(self, in_planes, planes, stride=2):
        super(BasicBlock, self).__init__()
        # First convolutional layer with specified stride
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, stride=stride, bias=False)
        # Batch normalization layer following the first convolution
        self.bn1 = nn.BatchNorm2d(planes)
        # ReLU activation function after the first batch normalization
        self.relu1 = nn.ReLU()
        # Second convolutional layer with a stride of 1
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, stride=1, bias=False)
        # Batch normalization layer following the second convolution
        self.bn2 = nn.BatchNorm2d(planes)
        # ReLU activation function after the second batch normalization
        self.relu2 = nn.ReLU()

    def forward(self, x):
        # Forward pass through the first set of layers
        out = self.relu1(self.bn1(self.conv1(x)))
        # Forward pass through the second set of layers
        out = self.relu1((self.bn2(self.conv2(out))))
        return out


class CNNModel(nn.Module):
    def __init__(self, block, num_blocks, num_classes=99):
        super(CNNModel, self).__init__()
        # Initial number of input planes for the first convolutional layer
        self.in_planes = 32

        # First convolutional layer of the model
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False)
        # Batch normalization layer following the first convolution of the model
        self.bn1 = nn.BatchNorm2d(32)
        # Layers created by stacking blocks
        self.layer1 = self._make_layer(block, 32, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 64, num_blocks[1], stride=1)
        self.layer3 = self._make_layer(block, 128, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 256, num_blocks[3], stride=2)
        # Final fully connected layer that outputs to the number of classes
        self.linear = nn.Linear(256 * block.expansion, num_classes)

        # Initialization of convolutional and batch normalization layers
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, num_blocks, stride):
        # Helper function to create a sequence of blocks with adjusted strides
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, out_feature=False):
        # Main forward pass through the network
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        feature = out.view(out.size(0), -1)
        out = self.linear(feature)
        if out_feature == False:
            return F.log_softmax(out, dim=1)
        else:
            return F.log_softmax(out, dim=1), feature


def main():
    best_test_loss = float('inf')
    patience_counter = 0
    n_epochs_stop = 40 # Stopped early when n epochs did not improve.

    # Calculate the time.
    start_time = time.time()

    # Calling the GPU.
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    torch.cuda.empty_cache()

    # Initialise the variable.
    n_epochs = 200 # Number of training sessions.
    batch_size_train = 512  # Training batch_size
    batch_size_test = 1024  # Testing batch_size
    # learning_rate = 0.001  # Learning rate. It's now set up in the optimiser.
    # momentum = 0.5 # When using SGD optimisation. # Solving the problem of large update swing of mini-batch SGD optimisation algorithm during gradient descent makes the convergence faster.
    log_interval = 10  # Operating Interval.
    random_seed = 2  # Random seed, set to get stable random numbers. Mainly for experimental replication.
    torch.manual_seed(random_seed)


    transform_train = transforms.Compose([
      transforms.RandomResizedCrop(32, scale=(0.8, 1.0)),
      transforms.RandomHorizontalFlip(),
      transforms.ToTensor(),
      transforms.Normalize((0.485, 0.456, 0.406),
                  (0.229, 0.224, 0.225)),
    ])

    transform_test  = transforms.Compose([
      transforms.Resize(32),
      transforms.ToTensor(),
      transforms.Normalize((0.485, 0.456, 0.406),
                  (0.229, 0.224, 0.225)),
    ])

    train_loader = DataLoader(
      datasets.ImageFolder(f'{DATA_ROOT}/train', transform=transform_train),
      batch_size=batch_size_train, shuffle=True, num_workers=4, pin_memory=True)

    test_loader  = DataLoader(
      datasets.ImageFolder(f'{DATA_ROOT}/test',  transform=transform_test),
      batch_size=batch_size_test, shuffle=False, num_workers=4, pin_memory=True)

    # Load the test set with enumerate.
    examples = enumerate(test_loader)

    # Get a batch.
    batch_idx, (example_data, example_targets) = next(examples)

    num_classes = len(train_loader.dataset.classes)

    # Instantiate a network.
    network = CNNModel(BasicBlock, [2,2,2,2], num_classes=num_classes)

    # Initialising the network
    network.apply(weight_init)

    # network.load_state_dict(torch.load('model_best.pth'))

    network.to((device))
    # Call the weights initialisation function.
    # Setting up the optimiser.
    # optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4)
    # optimizer = optim.Adam(network.parameters(), lr=learning_rate)
    # optimizer = optim.RMSprop(network.parameters(),lr=learning_rate,alpha=0.99,momentum = momentum)
    optimizer = optim.Adam(network.parameters(), lr=0.001, weight_decay=1e-4)

    # Set the learning rate gradient to decrease if the accuracy does not increase for three consecutive epoch tests.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

    # Defines a list of stored data.
    train_losses = []
    train_counter = []
    train_acces = []
    test_losses = []
    test_counter = [i * len(train_loader.dataset) for i in range(n_epochs + 1)]
    test_acces = []
    true_labels = []
    predicted_labels = []

    # Looking first at the recognition ability of the model, it can be seen that the untrained model performs poorly on the test set, with roughly only about 10% correct recognition.
    test(1, network, test_loader, device, test_acces, test_losses, optimizer, true_labels, predicted_labels)

    ### Training and tested after each epoch. ###
    ###################################################
    # Formal training based on the number of epochs and testing at the end of each epoch.
    for epoch in range(1, n_epochs + 1):
        train(epoch, network, train_loader, optimizer, device, train_losses, train_counter, train_acces, log_interval)
        test(epoch, network, test_loader, device, test_acces, test_losses, optimizer, true_labels, predicted_labels)
        scheduler.step()  # Called at the end of each epoch.

        # Check for improvements.
        current_test_loss = test_losses[-1]
        if current_test_loss < best_test_loss:
            best_test_loss = current_test_loss
            patience_counter = 0
            # Saves the current best model state.
            best_model_state = network.state_dict()
        else:
            patience_counter += 1

        if patience_counter >= n_epochs_stop:
            print(f'Early stopping at epoch {epoch} with test loss {best_test_loss}')
            # Load the best saved model state.
            network.load_state_dict(best_model_state)
            break

    # Preserve the best model at early stops.
    torch.save(best_model_state, './model_best.pth')

    # Enter the accuracy of the last saved model, which is the maximum test accuracy.
    print('\n\033[1;31mThe network Max Avg Accuracy : {:.2f}%\033[0m'.format(100. * max(test_acces)))
    report = classification_report(true_labels, predicted_labels, digits=2, output_dict=True, zero_division=1)
    print("              precision    recall  f1-score   support\n")
    for label, metrics in report.items():
        if label.isdigit():  # Only for numeric labels
            print(
                f"          {label}       {metrics['precision']:.2f}      {metrics['recall']:.2f}      {metrics['f1-score']:.2f}      {int(metrics['support'])}")
    print(f"\n    accuracy                           {report['accuracy']:.2f}     {len(true_labels)}")
    print(
        f"   macro avg       {report['macro avg']['precision']:.2f}      {report['macro avg']['recall']:.2f}      {report['macro avg']['f1-score']:.2f}     {len(true_labels)}")
    print(
        f"weighted avg       {report['weighted avg']['precision']:.2f}      {report['weighted avg']['recall']:.2f}      {report['weighted avg']['f1-score']:.2f}     {len(true_labels)}")
    total_time = time.time() - start_time
    print(f"\nTime cost: {total_time // 60}mins")

    # visualisation
    fig = plt.figure(figsize=(15,5))  # Enlarge the drawing window 15 times horizontally and 5 times vertically.
    ax1 = fig.add_subplot(121)
    # Training Losses
    ax1.plot(train_counter, train_losses, color='blue')
    # Test Losses
    if len(test_counter) > len(test_losses):
        test_counter = test_counter[:len(test_losses)]

    ax1.scatter(test_counter, test_losses, color='red')
    # illustration
    ax1.legend(['Train Loss', 'Test Loss'], loc='upper right')
    ax1.set_title('Train & Test Loss')
    ax1.set_xlabel('number of training examples seen')
    ax1.set_ylabel('negative log likelihood loss')

    # Accuracy curve
    ax2 = fig.add_subplot(122)
    # Training accuracy
    actual_epochs_trained = len(train_acces)
    ax2.plot(range(1, actual_epochs_trained + 1), train_acces, 'b-', label='Train acc')
    # Raw model test accuracy
    ax2.plot(range(1, len(test_acces) + 1), test_acces, 'r-', label='Original Test acc')

    # Mark the point of maximum test accuracy
    max_test_acces_epoch = test_acces.index(max(test_acces))
    max_test_acces_value = round(max(test_acces), 4)
    ax2.plot(max_test_acces_epoch, max_test_acces_value, 'ko')  # maximum point

    show_max = f'[{max_test_acces_epoch}, {max_test_acces_value}]'
    # Maximum point coordinates display
    ax2.annotate(show_max, xy=(max_test_acces_epoch, max_test_acces_value),
                xytext=(max_test_acces_epoch, max_test_acces_value))

    # illustration
    ax2.legend()
    ax2.set_title('Train & Test Accuracy')
    ax2.set_xlabel('number of training epoch')
    ax2.set_ylabel('Accuracy')
    plt.show()

if __name__ == '__main__':
    main()
