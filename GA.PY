import os
import time

import numpy as np
from matplotlib import pyplot as plt
!pip uninstall -y scikit-opt
!pip install scikit-opt==0.6.3   
from sko.GA import GA             

import torch
import torch.nn as nn
import torch.nn.functional as F
!pip install -U torch-pruning
import torch_pruning as tp
from torchvision import transforms, datasets
from torch.utils.data import DataLoader



DATA_ROOT = '/content/ycb_rgb256'   # ImageFolder root
MODEL_PATH = 'model_best.pth'   # Weights saved in the training script
IMG_SIZE   = 32                    # Side length after offline scaling
NUM_CLASS = len(datasets.ImageFolder(f'{DATA_ROOT}/train').classes)


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

# Import the Block in the model.
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=2):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=True)   

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=True)   # Second ReLU (new)

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))  # No longer reuse relu1
        return out



device = "cuda:0"
# network = CNNModel(BasicBlock, [2, 2, 2], 10)

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
torch.backends.cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()

# Initialising variables.
n_epochs = 1000
batch_size_train = 512
batch_size_test = 1024
random_seed = 2
torch.manual_seed(random_seed)


transform_test = transforms.Compose([
    transforms.Resize(32),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406),
                         (0.229, 0.224, 0.225)),
])

test_loader = DataLoader(
    datasets.ImageFolder(f'{DATA_ROOT}/test',  transform=transform_test),
    batch_size=1024, shuffle=False, num_workers=4, pin_memory=True)

def build_model():
    net = CNNModel(BasicBlock, [2,2,2,2], num_classes=NUM_CLASS)
    net.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
    return net

# Getting Accuracy.
def get_acc(network):
    global device
    network.to(device)

    s = time.time()
    true_labels = []
    predicted_labels = []
    network.eval()  # Set the network to evaluating mode.
    test_loss = 0
    correct = 0
    test_acces=[]
    test_losses=[]
    with torch.no_grad():
        for data, target in test_loader:
            output = network(data.to(device))  # Pass in this set of batch for forward computation.
            # test_loss += F.nll_loss(output, target, size_average=False).item()
            true_labels.extend(target.tolist())
            predicted_labels.extend(output.argmax(dim=1).tolist())

            test_loss += F.nll_loss(output, target.to(device), reduction='sum').item()

            pred = output.data.max(dim=1, keepdim=True)[1]  # Take the largest category in the output.
            # dim = 1 means to go to the maximum of each row, [1] means to take the index of the maximum without going to the maximum itself [0].

            correct += pred.eq(target.data.view_as(pred).to(device)).sum()  # Compare and find the number of correct classifications.
    acc = correct / len(test_loader.dataset)  # Average test accuracy.
    e = time.time()
    if test:
        device = torch.device("cpu")  # Explicitly set the device to CPU for this test
        network.to(device)

        print("time_usage:" ,e-s)
        test_acces.append(acc.item())

        test_loss /= len(test_loader.dataset)  # The average loss, len, is 10000.
        test_losses.append(test_loss)  # Records the test_loss for the epoch.

        print('\r Test set \033[1;31m\033[0m : Avg. loss: {:.4f}, Accuracy: {}/{}  \033[1;31m({:.2f}%)\033[0m\n' \
              .format( test_loss, correct, len(test_loader.dataset), 100. * acc), end='')
    return acc.cpu().item()


# Define a sequence of transformations to apply to the train images.
transform_train = transforms.Compose([
    transforms.RandomResizedCrop(32, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406),
                         (0.229, 0.224, 0.225)),
])

# YCB-RGB32 train loader
train_loader = DataLoader(
    datasets.ImageFolder(f'{DATA_ROOT}/train', transform=transform_train),
    batch_size=512, shuffle=True, num_workers=4, pin_memory=True)

test=False

def bn_recalib(net, loader, num_batches=200, device=device):
    """
    Recalibration BatchNorm running_mean / running_var
    ------------------------------------------------
    Parameters:
        net         : Pruned network
        loader      : DataLoader for traversing data 
        num_batches : How many batches to use to estimate the statistics
        device      : Device, same as global device
    """
    net.train()                    # BN statistics are only updated in train() mode.
    with torch.no_grad():
        for i, (x, _) in enumerate(loader):
            net(x.to(device))
            if i + 1 == num_batches:
                break
    net.eval()                     # Switch back to eval(), for subsequent reasoning.
# ========================================================================


def obj_func(x):
    global device
    # Load the trained model.
    network = build_model().to(device)
    if test:
        device = torch.device("cpu")  # Set the device to CPU for this test
        network.to(device)

        print("Before Pruning：")
        get_acc(network)
    # Prepare virtual inputs for pruning, modelling the dimensions and types of normal input data.
    example_inputs = torch.randn(10, 3, 32, 32)
    example_inputs = example_inputs.to(device)

    # Define a dictionary that maps the names of pruning policies to their corresponding implementations.
    # x[1] is dynamically passed in, for L1 The pruning strategy denotes the order of the paradigm.
    imp_dict = {
        'Group Hessian': tp.importance.HessianImportance(group_reduction='mean'),  # Use the Hessian matrix and take the mean of the importance within the group.
        'Single-layer Hessian': tp.importance.HessianImportance(group_reduction='first'),  # Use the Hessian matrix, but consider only the first layer.

        'Group Taylor': tp.importance.TaylorImportance(group_reduction='mean'),  # Use the first-order derivative of the Taylor expansion and take the mean of the importance within the group.
        'Single-layer Taylor': tp.importance.TaylorImportance(group_reduction='first'),  # Use the first order derivatives of the Taylor expansion, but consider only the first layer.

        'Group L1': tp.importance.MagnitudeImportance(p=int(x[1]), group_reduction='mean'),  # L1 norms were used and averaged for within-group significance.
        'Single-layer L1': tp.importance.MagnitudeImportance(p=int(x[1]), group_reduction='first'),  # L1 norms are used, but only the first layer is considered.

        'Group Slimming': tp.importance.BNScaleImportance(group_reduction='mean'),  # Scaling factors for the BN layer were used and averaged for within-group importance.
        'Single-layer Slimming': tp.importance.BNScaleImportance(group_reduction='first'),  # Use the scaling factor of the BN layer, but only consider the first layer.

        'Random': tp.importance.RandomImportance(),  # Randomly selected weights for pruning.
    }

    # Selection of pruning strategy, x[0] is dynamically passed in to select the corresponding pruning strategy.
    imp = list(imp_dict.values())[int(x[0])]

    # Special treatment for pruning strategies that use Hessian matrices.
    if isinstance(imp, tp.importance.HessianImportance):
        for batch_idx, (data, target) in enumerate(train_loader):
            if batch_idx > 10: break
            output = network(data.to(device))
            loss = F.nll_loss(output, target.to(device))
            imp.zero_grad()  # clear accumulated gradients
            loss.backward()

    if isinstance(imp, tp.importance.TaylorImportance):
        for batch_idx, (data, target) in enumerate(train_loader):
            if batch_idx > 10: break
            output = network(data.to(device))
            loss = F.nll_loss(output, target.to(device))
            loss.backward()

    # Identify layers that are not involved in pruning, e.g., the final fully connected layer.
    ignored_layers = []
    for m in network.modules():
        if isinstance(m, torch.nn.Linear):
            ignored_layers.append(m)

    # Define the pruner with the pruning strategy, pruning ratio, and layers to ignore. (The pruner can also be used as a population-if-evolved algorithm if desired, but this will require more iterations.)
    pruner = tp.pruner.MetaPruner(
        network,
        example_inputs,
        importance=imp,
        pruning_ratio=x[2] / 1000,  # x[2] Dynamic incoming pruning ratio.
        ignored_layers=ignored_layers,


    # Record the network structure and number of parameters before pruning.
    base_macs, base_nparams = tp.utils.count_ops_and_params(network, example_inputs)

    # Perform pruning.
    pruner.step()
    bn_recalib(network, train_loader, num_batches=100, device=device)

    # Record the network structure and number of parameters after pruning.
    macs, nparams = tp.utils.count_ops_and_params(network, example_inputs)
    if test:
        device = torch.device("cpu")  # Set the device to CPU for this test
        network.to(device)

        print("After Pruning：")
        torch.save(network.state_dict(), 'pruned_network.pth')
        acc = float(get_acc(network))

        return
    # Calculate the accuracy of the model after pruning.
    acc = float(get_acc(network))

    #
    target_acc = 0.90
    penalty  = max(0., target_acc - acc)

    pruner_names = [
        "GrowingRegPruner",
        "MetaPruner",
        "BNScalePruner",
        "GroupNormPruner"
    ]
    print("==================" * 20)
    print("Pruning Strategy：", list(imp_dict.keys())[int(x[0])])
    # print("Pruners for:", pruner_names[int(x[4])])
    print("When the pruning policy is Group L1 or Single-layer L1 Parameter1 ", x[1])
    print("Pruning Rate ", x[2] / 1000)
    print(f"Number of parameters after pruning：{base_nparams} => {nparams}")
    print(f"Accuracy after pruning：{0.95} => {acc}")
    # print(f"round_to：{x[3]}")
    print("x:",x)

    reduce = float(nparams / base_nparams)  # Calculate the percentage reduction in parameters after model pruning.
    a = 0.5  # The larger the model lightweight weight , the smaller the number of end result model parameters.
    b = 0.5  # Model accuracy weights, the greater the final result accuracy a+b=1.

    # Return value checking.
    value = a * reduce - b * acc + penalty * 200  # <<< 50 为罚分系数，可调
    print("value:", value)
    return value


def fig():
    network = build_model().to(device)

    # Virtual input for pruning methods.
    example_inputs = torch.randn(10, 3, 32, 32)
    example_inputs = example_inputs.to(device)

    # Prepare pruning ratio ranges and results storage.
    pruning_ratios = np.linspace(0.01, 0.2, 20)  # From 0.01 to 1 pruning ratio.
    strategy_results = {}

    for strategy_id in [0, 1, 2, 3, 4, 5, 6, 7, 8]:  # Iterative pruning strategies.
        print(strategy_id)
        accuracies = []
        for ratio in pruning_ratios:
            # Reload the model state to avoid the effects of the previous pruning.
            network = build_model().to(device)
            imp_dict = {
                'Group Hessian': tp.importance.HessianImportance(group_reduction='mean'),
                'Single-layer Hessian': tp.importance.HessianImportance(group_reduction='first'),

                'Group Taylor': tp.importance.TaylorImportance(group_reduction='mean'),
                'Single-layer Taylor': tp.importance.TaylorImportance(group_reduction='first'),

                'Group L1': tp.importance.MagnitudeImportance(p=1, group_reduction='mean'),
                'Single-layer L1': tp.importance.MagnitudeImportance(p=1, group_reduction='first'),

                'Group Slimming': tp.importance.BNScaleImportance(group_reduction='mean'),
                'Single-layer Slimming': tp.importance.BNScaleImportance(group_reduction='first'),

                'Random': tp.importance.RandomImportance(),
            }
            imp = list(imp_dict.values())[strategy_id]

            # Ignore specific layers.
            ignored_layers = []
            for m in network.modules():
                if isinstance(m, torch.nn.Linear):
                    ignored_layers.append(m)

            if isinstance(imp, tp.importance.HessianImportance):
                for batch_idx, (data, target) in enumerate(train_loader):
                    if batch_idx > 10: break
                    output = network(data.to(device))
                    loss = F.nll_loss(output, target.to(device))
                    imp.zero_grad()  # clear accumulated gradients
                    loss.backward()

            if isinstance(imp, tp.importance.TaylorImportance):
                for batch_idx, (data, target) in enumerate(train_loader):
                    if batch_idx > 10: break
                    output = network(data.to(device))
                    loss = F.nll_loss(output, target.to(device))
                    loss.backward()

            # Configure pruners and perform pruning.
            pruner = tp.pruner.GroupNormPruner(
                network,
                example_inputs,
                importance=imp,
                pruning_ratio=ratio,
                ignored_layers=ignored_layers
            )
            pruner.step()

            # Calculate and record accuracy.
            acc = float(get_acc(network))
            accuracies.append(acc)

        # Stores the results of the current strategy.
        strategy_results[strategy_id] = accuracies
    print(strategy_results)

    # Plot result
    strategy_names = {
    0: 'Group Hessian',
    1: 'Single-layer Hessian',
    2: 'Group Taylor',
    3: 'Single-layer Taylor',
    4: 'Group L1',
    5: 'Single-layer L1',
    6: 'Group Slimming',
    7: 'Single-layer Slimming',
    8: 'Random'
    }

    for strategy_id, accs in strategy_results.items():
        plt.plot(pruning_ratios, accs, label=strategy_names[strategy_id])

    plt.xlabel("Pruning Ratio")
    plt.ylabel("Accuracy")
    plt.title("Pruning Strategy Comparison")
    plt.legend()
    plt.show()

if __name__ == '__main__':
    # fig()
    # Genetic Algorithm Definitions: n_dim:Dimension of the decision variable. lb ub Upper and lower bounds of the decision variable. prob_mut Variability rate, size_pop Population size max_iter Maximum number of iterations. precision Precision of the decision variable, which is 1 for an integer.
    ga = GA(func=obj_func, n_dim=3, lb=[0, 0, 10], ub=[7, 8, 200], prob_mut=0.01, size_pop=10,
            max_iter=10, precision=[1, 1, 1])
    best_x, best_y = ga.run()
    print("The optimal pruning strategy is as follows：")
    print("=====================================" * 10)
    print(best_x, best_y)
    obj_func(best_x)
    test=True
    obj_func(best_x)