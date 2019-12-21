import torch
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
import torch.nn as nn
from hessian_eigenthings.lanczos_upd import lanczos


# ---------------------------------- Hessian-vector product operator----------------------------------

class Operator():
    def __init__(self, size):
        self._size = size

    def apply(self, vec):
        raise NotImplementedError()

    def __matmul__(self, vec):
        return self.apply(vec)

    def size(self):
        return self._size


class ModelHessianOperator(Operator):
    def __init__(self, model, criterion, data_input, data_target):
        size = int(sum(p.numel() for p in model.parameters()))
        super(ModelHessianOperator, self).__init__(size)
        self._model = model
        self._criterion = criterion
        self.set_model_data(data_input, data_target)

    def apply(self, vec):
        return to_vector(torch.autograd.grad(self._grad, self._model.parameters()
                                             , grad_outputs=vec, only_inputs=True, retain_graph=True))

    def set_model_data(self, data_input, data_target):
        self._data_input = data_input
        self._data_target = data_target
        self._output = self._model(self._data_input)
        self._loss = self._criterion(self._output, self._data_target)
        self._grad = to_vector(torch.autograd.grad(self._loss, self._model.parameters(), create_graph=True))

    def get_input(self):
        return self._data_input

    def get_target(self):
        return self._data_target


def to_vector(tensors):
    return torch.cat([t.contiguous().view(-1) for t in tensors])


# ----------------------------------------- Training parameters -----------------------------------------

batch_size = 200

train_dataset = MNIST(root='MNIST', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = MNIST(root='MNIST', train=False, transform=transforms.ToTensor(), download=True)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

n_iters = 500
epochs = n_iters / (len(train_dataset) / batch_size)
input_dim = 784
output_dim = 10
lr_rate = 0.001


# ----------------------------------------- Logistic Regression -----------------------------------------

# class LogisticRegression(torch.nn.Module):
#     def __init__(self, input_dim, output_dim):
#         super(LogisticRegression, self).__init__()
#         self.linear = torch.nn.Linear(input_dim, output_dim)
#
#     def forward(self, x):
#         outputs = self.linear(x)
#         return outputs
#
#
# model = LogisticRegression(input_dim, output_dim)
# criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.SGD(model.parameters(), lr=lr_rate)
#
# iter_num = 0
# for epoch in range(int(epochs)):
#     for i, (images, labels) in enumerate(train_loader):
#         images = Variable(images.view(-1, 28 * 28))
#         labels = Variable(labels)
#
#         optimizer.zero_grad()
#         outputs = model(images)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()
#
#         iter_num+=1
#         if iter_num % 500==0:
#             # calculate Accuracy
#             correct = 0
#             total = 0
#             for images, labels in test_loader:
#                 images = Variable(images.view(-1, 28*28))
#                 outputs = model(images)
#                 _, predicted = torch.max(outputs.data, 1)
#                 total+= labels.size(0)
#                 # for gpu, bring the predicted and labels back to cpu fro python operations to work
#                 correct+= (predicted == labels).sum()
#             accuracy = 100 * correct/total
#             print("Iteration: {}. Loss: {}. Accuracy: {}%.".format(iter_num, loss.item(), accuracy))
#
# test_batch = next(iter(test_loader))
# test_batch[0] = test_batch[0].view(-1, 28 * 28)
#
# data_input = test_batch[0]
# data_target = test_batch[1]
# op = ModelHessianOperator(model, criterion, data_input, data_target)
# size = to_vector(model.parameters()).shape[0]


# ----------------------------------------- ConvNN -----------------------------------------

model = nn.Sequential(
    nn.Conv2d(1, 8, kernel_size=3),
    nn.ReLU(),
    nn.Conv2d(8, 8, kernel_size=3),
    nn.ReLU(),

    nn.MaxPool2d(2),

    nn.Conv2d(8, 16, kernel_size=3),
    nn.ReLU(),
    nn.Conv2d(16, 16, kernel_size=3),
    nn.ReLU(),

    nn.MaxPool2d(2),
    nn.Flatten(),
    nn.Linear(256, 10),
)

# model = model.cuda()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=lr_rate)

lr_rate = 0.00007
iter_num = 0
for epoch in range(int(epochs)):
    for i, (images, labels) in enumerate(train_loader):
        # images = images.cuda()
        # labels = labels.cuda()
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        iter_num+=1
        if iter_num % 500==0:
            # calculate Accuracy
            correct = 0
            total = 0
            for images, labels in test_loader:
                # images = images.cuda()
                # labels = labels.cuda()
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total+= labels.size(0)
                # for gpu, bring the predicted and labels back to cpu fro python operations to work
                correct+= (predicted == labels).sum()
            accuracy = 100 * correct/total
            print("Iteration: {}. Loss: {}. Accuracy: {}%.".format(iter_num, loss.item(), accuracy))

test_batch = next(iter(test_loader))
data_input = test_batch[0]
data_target = test_batch[1]
# op = ModelHessianOperator(model, criterion, data_input.cuda(), data_target.cuda())
op = ModelHessianOperator(model, criterion, data_input, data_target)
size = to_vector(model.parameters()).shape[0]


# ----------------------------------------- Lanczoc -----------------------------------------

print('The model has been trained')
print('Starting Lanczoc method')
num_lanczos_vectors = int(0.5 * size)
T, V = lanczos(operator=op, num_lanczos_vectors=num_lanczos_vectors, size=size, use_gpu=False)
print(T)
print(V)



