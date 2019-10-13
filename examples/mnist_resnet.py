from torchvision.models.resnet import ResNet, BasicBlock, Bottleneck
from torchvision.datasets import MNIST
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import inspect
import time
from torch import nn, optim
import torch
from torchvision.transforms import Compose, ToTensor, Normalize, Resize
from torch.utils.data import DataLoader


class MnistResNet(ResNet):
    def __init__(self):
        super(MnistResNet, self).__init__(BasicBlock, [2, 2, 2, 2], num_classes=10)
        self.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if x.is_quantized:
            x = x.dequantize()

        x = self.avgpool(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)
        return torch.softmax(x, dim=-1)


def get_data_loaders(train_batch_size, val_batch_size):
    mnist = MNIST(download=True, train=True, root=".").train_data.float()

    data_transform = Compose([Resize((224, 224)), ToTensor(), Normalize((mnist.mean() / 255,), (mnist.std() / 255,))])

    train_loader = DataLoader(MNIST(download=True, root=".", transform=data_transform, train=True),
                              batch_size=train_batch_size, shuffle=True)

    val_loader = DataLoader(MNIST(download=False, root=".", transform=data_transform, train=False),
                            batch_size=val_batch_size, shuffle=False)
    return train_loader, val_loader


def calculate_metric(metric_fn, true_y, pred_y):
    if "average" in inspect.getfullargspec(metric_fn).args:
        return metric_fn(true_y, pred_y, average="macro")
    else:
        return metric_fn(true_y, pred_y)


def print_scores(p, r, f1, a, batch_size):
    for name, scores in zip(("precision", "recall", "F1", "accuracy"), (p, r, f1, a)):
        print(f"\t{name.rjust(14, ' ')}: {sum(scores) / batch_size:.4f}")

def do_train():
    start_ts = time.time()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epochs = 1

    model = MnistResNet().to(device)
    train_loader, val_loader = get_data_loaders(64, 64)

    losses = []
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adadelta(model.parameters())

    batches = len(train_loader)
    val_batches = len(val_loader)

    # training loop + eval loop
    for epoch in range(epochs):
        total_loss = 0
        model.train()

        for i, data in enumerate(train_loader):
            X, y = data[0].to(device), data[1].to(device)

            model.zero_grad()
            outputs = model(X)
            loss = loss_function(outputs, y)

            loss.backward()
            optimizer.step()
            current_loss = loss.item()
            total_loss += current_loss
            print("{}/{}Loss: {:.4f}".format(i, len(train_loader), total_loss / (i + 1)))

        torch.cuda.empty_cache()

        val_losses = 0
        precision, recall, f1, accuracy = [], [], [], []

        model.eval()
        with torch.no_grad():
            for i, data in enumerate(val_loader):
                X, y = data[0].to(device), data[1].to(device)
                outputs = model(X)
                val_losses += loss_function(outputs, y)

                predicted_classes = torch.max(outputs, 1)[1]

                for acc, metric in zip((precision, recall, f1, accuracy),
                                       (precision_score, recall_score, f1_score, accuracy_score)):
                    acc.append(
                        calculate_metric(metric, y.cpu(), predicted_classes.cpu())
                    )

        print(
            f"Epoch {epoch + 1}/{epochs}, training loss: {total_loss / batches}, validation loss: {val_losses / val_batches}")
        print_scores(precision, recall, f1, accuracy, val_batches)
        losses.append(total_loss / batches)
    torch.save(model.cpu().state_dict(), "mnist_resnet.pt")
    print(losses)
    print(f"Training time: {time.time() - start_ts}s")


from truckms.inference.quantization import *
from torch.quantization.fuse_modules import fuse_modules
from torch import quantize_per_tensor

def do_quantization():
    device = 'cpu'
    model = MnistResNet().to(device)
    model.load_state_dict(torch.load("mnist_resnet.pt"))
    train_loader, val_loader = get_data_loaders(256, 256)

    val_batches = len(val_loader)

    modules_to_fuse = get_modules_to_fuse(model)
    replace_frozenbatchnorm_batchnorm(model)
    model.eval()
    fuse_modules(model, modules_to_fuse, inplace=True, fuser_func=custom_fuse_func)

    from torch.quantization.QConfig import default_qconfig
    from torch.quantization.default_mappings import DEFAULT_MODULE_MAPPING
    from torch.quantization.quantize import prepare, propagate_qconfig_
    import torch.nn.intrinsic as nni
    import itertools

    for child in model.modules():
        if isinstance(child, nn.ReLU):
            child.inplace = False

    # TODO i removed the linear layers because they were too complicated for quantization. too much logic
    # TODO maybe have alook here https://github.com/pytorch/pytorch/files/2994852/Model.Quantization.for.Pytorch.pdf
    qconfig_spec = dict(zip({nn.Conv2d, nni.ConvReLU2d, nn.ReLU}, itertools.repeat(default_qconfig)))
    propagate_qconfig_(model, qconfig_spec)
    model.eval()

    local_Data =None
    for i, data in enumerate(val_loader):
        X, y = data[0].to(device), data[1].to(device)
        break

    def run_fn(model, run_agrs):
        return model(X)
    model = torch.quantization.quantize(model, run_fn=run_fn, run_args={}, mapping=DEFAULT_MODULE_MAPPING)

    print (model)

    model.eval()
    precision, recall, f1, accuracy = [], [], [], []

    with torch.no_grad():
        for i, data in enumerate(val_loader):
            X, y = data[0].to(device), data[1].to(device)
            X = quantize_per_tensor(X, 1 / (2 ** 8), 0, torch.quint8)
            outputs = model(X)


            predicted_classes = torch.max(outputs, 1)[1]

            for acc, metric in zip((precision, recall, f1, accuracy),
                                   (precision_score, recall_score, f1_score, accuracy_score)):
                acc.append(
                    calculate_metric(metric, y.cpu(), predicted_classes.cpu())
                )

    print_scores(precision, recall, f1, accuracy, val_batches)


if __name__ == "__main__":
    # do_train()
    do_quantization()