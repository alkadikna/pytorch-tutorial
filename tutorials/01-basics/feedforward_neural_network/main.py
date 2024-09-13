import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.optim as optim


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters 
input_size = 784
hidden_size = 500
num_classes = 10
num_epochs = 5
batch_size = 100
learning_rate = 0.001

# MNIST dataset 
train_dataset = torchvision.datasets.MNIST(root='../../data', 
                                           train=True, 
                                           transform=transforms.ToTensor(),  
                                           download=True)

test_dataset = torchvision.datasets.MNIST(root='../../data', 
                                          train=False, 
                                          transform=transforms.ToTensor())

# # Data loader
# train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
#                                            batch_size=batch_size, 
#                                            shuffle=True)

# test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
#                                           batch_size=batch_size, 
#                                           shuffle=False)

# Fully connected neural network with one hidden layer
# Definisikan arsitektur Neural Network dengan modifikasi jumlah hidden layers dan neurons
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, num_classes):
        super(NeuralNet, self).__init__()
        # Fully connected layer 1
        self.fc1 = nn.Linear(input_size, hidden_size1)
        # Activation function ReLU
        self.relu1 = nn.ReLU()
        # Fully connected layer 2 (hidden layer tambahan)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        # Activation function Sigmoid
        self.sigmoid = nn.Sigmoid()
        # Output layer
        self.fc3 = nn.Linear(hidden_size2, num_classes)

    def forward(self, x):
        # Forward pass: Input ke hidden layer 1
        out = self.fc1(x)
        out = self.relu1(out)
        # Forward pass: Hidden layer 1 ke hidden layer 2
        out = self.fc2(out)
        out = self.sigmoid(out)
        # Forward pass: Hidden layer 2 ke output layer
        out = self.fc3(out)
        return out
    
# class NeuralNet2(nn.Module):
#     def __init__(self, input_size, hidden_size1, hidden_size2, num_classes):
#         super(NeuralNet2, self).__init__()
#         # Fully connected layer 1
#         self.fc1 = nn.Linear(input_size, hidden_size1)
#         # Activation function ReLU
#         self.relu1 = nn.ReLU()
#         # Fully connected layer 2 (hidden layer tambahan)
#         self.fc2 = nn.Linear(hidden_size1, hidden_size2)
#         # Output layer
#         self.fc3 = nn.Linear(hidden_size2, num_classes)

#     def forward(self, x):
#         # Forward pass: Input ke hidden layer 1
#         out = self.fc1(x)
#         out = self.relu1(out)
#         # Forward pass: Hidden layer 1 ke hidden layer 2
#         out = self.fc2(out)
#         out = self.relu1(out)
#         # Forward pass: Hidden layer 2 ke output layer
#         out = self.fc3(out)
#         return out
    
# class NeuralNet3(nn.Module):
#     def __init__(self, input_size, hidden_size1, hidden_size2, num_classes):
#         super(NeuralNet3, self).__init__()
#         # Fully connected layer 1
#         self.fc1 = nn.Linear(input_size, hidden_size1)
#         # Activation function ReLU
#         self.sigmoid = nn.Sigmoid()
#         # Fully connected layer 2 (hidden layer tambahan)
#         self.fc2 = nn.Linear(hidden_size1, hidden_size2)
#         # Output layer
#         self.fc3 = nn.Linear(hidden_size2, num_classes)

#     def forward(self, x):
#         # Forward pass: Input ke hidden layer 1
#         out = self.fc1(x)
#         out = self.sigmoid(out)
#         # Forward pass: Hidden layer 1 ke hidden layer 2
#         out = self.fc2(out)
#         out = self.sigmoid(out)
#         # Forward pass: Hidden layer 2 ke output layer
#         out = self.fc3(out)
#         return out

# # Inisialisasi model dengan ukuran input, hidden layers, dan jumlah kelas
# model = NeuralNet(input_size=784, hidden_size1=128, hidden_size2=64, num_classes=10).to(device)
# # model2 = NeuralNet2(input_size=784, hidden_size1=128, hidden_size2=64, num_classes=10).to(device)
# # model3 = NeuralNet3(input_size=784, hidden_size1=128, hidden_size2=64, num_classes=10).to(device)

# # Definisikan fungsi loss (CrossEntropyLoss digunakan untuk klasifikasi)
# criterion = nn.CrossEntropyLoss()
# # Definisikan optimizer (Adam digunakan untuk update weights)
# optimizer = optim.Adam(model.parameters(), lr=0.001)
# # optimizer2 = optim.Adam(model2.parameters(), lr=0.001)
# # optimizer3 = optim.Adam(model3.parameters(), lr=0.001)

# # Untuk menyimpan nilai loss setiap iterasi
# loss_list = []
# # loss_list2 = []
# # loss_list3 = []

# # Training the model
# total_step = len(train_loader)
# for epoch in range(num_epochs):
#     for i, (images, labels) in enumerate(train_loader):
#         # Ubah dimensi gambar dan pindahkan ke device (CPU/GPU)
#         images = images.reshape(-1, 28 * 28).to(device)
#         labels = labels.to(device)

#         # Forward pass: Hitung prediksi dari model
#         outputs = model(images)
#         # Hitung loss berdasarkan prediksi dan label sebenarnya
#         loss = criterion(outputs, labels)

#         # Backward pass: Zero grad, backward, dan optimize
#         optimizer.zero_grad()  # Set gradien ke 0 sebelum backward
#         loss.backward()  # Hitung gradien
#         optimizer.step()  # Update weights berdasarkan gradien

#         # Simpan nilai loss
#         loss_list.append(loss.item())

#         # Print loss setiap 100 step
#         if (i + 1) % 100 == 0:
#             print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
#                   .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))
            
# # # Training the model 2
# # total_step = len(train_loader)
# # for epoch in range(num_epochs):
# #     for i, (images, labels) in enumerate(train_loader):
# #         # Ubah dimensi gambar dan pindahkan ke device (CPU/GPU)
# #         images = images.reshape(-1, 28 * 28).to(device)
# #         labels = labels.to(device)

# #         # Forward pass: Hitung prediksi dari model
# #         outputs = model2(images)
# #         # Hitung loss berdasarkan prediksi dan label sebenarnya
# #         loss2 = criterion(outputs, labels)

# #         # Backward pass: Zero grad, backward, dan optimize
# #         optimizer2.zero_grad()  # Set gradien ke 0 sebelum backward
# #         loss2.backward()  # Hitung gradien
# #         optimizer2.step()  # Update weights berdasarkan gradien

# #         # Simpan nilai loss
# #         loss_list2.append(loss2.item())

# #         # Print loss setiap 100 step
# #         if (i + 1) % 100 == 0:
# #             print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
# #                   .format(epoch + 1, num_epochs, i + 1, total_step, loss2.item()))
            
# # # Training the model 3
# # total_step = len(train_loader)
# # for epoch in range(num_epochs):
# #     for i, (images, labels) in enumerate(train_loader):
# #         # Ubah dimensi gambar dan pindahkan ke device (CPU/GPU)
# #         images = images.reshape(-1, 28 * 28).to(device)
# #         labels = labels.to(device)

# #         # Forward pass: Hitung prediksi dari model
# #         outputs = model3(images)
# #         # Hitung loss berdasarkan prediksi dan label sebenarnya
# #         loss3 = criterion(outputs, labels)

# #         # Backward pass: Zero grad, backward, dan optimize
# #         optimizer3.zero_grad()  # Set gradien ke 0 sebelum backward
# #         loss3.backward()  # Hitung gradien
# #         optimizer3.step()  # Update weights berdasarkan gradien

# #         # Simpan nilai loss
# #         loss_list3.append(loss3.item())

# #         # Print loss setiap 100 step
# #         if (i + 1) % 100 == 0:
# #             print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
# #                   .format(epoch + 1, num_epochs, i + 1, total_step, loss3.item()))

# # Visualisasi loss terhadap iterasi
# plt.figure(figsize=(10, 5))
# plt.plot(loss_list, label='Loss ReLu and Sigmoid')
# # plt.plot(loss_list2, label='Loss ReLu')
# # plt.plot(loss_list3, label='Loss Sigmoid')
# plt.xlabel('Iteration')
# plt.ylabel('Loss')
# plt.title('Loss Comparison')
# plt.legend()
# plt.grid(True)
# plt.show()

# # Test the model
# # In test phase, we don't need to compute gradients (for memory efficiency)
# with torch.no_grad():
#     correct = 0
#     total = 0
#     for images, labels in test_loader:
#         images = images.reshape(-1, 28*28).to(device)
#         labels = labels.to(device)
#         outputs = model(images)
#         _, predicted = torch.max(outputs.data, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()

#     print('Accuracy of the network on the 10000 test images model 1: {} %'.format(100 * correct / total))

# # with torch.no_grad():
# #     correct = 0
# #     total = 0
# #     for images, labels in test_loader:
# #         images = images.reshape(-1, 28*28).to(device)
# #         labels = labels.to(device)
# #         outputs = model2(images)
# #         _, predicted = torch.max(outputs.data, 1)
# #         total += labels.size(0)
# #         correct += (predicted == labels).sum().item()

# #     print('Accuracy of the network on the 10000 test images model 2: {} %'.format(100 * correct / total))

# # with torch.no_grad():
# #     correct = 0
# #     total = 0
# #     for images, labels in test_loader:
# #         images = images.reshape(-1, 28*28).to(device)
# #         labels = labels.to(device)
# #         outputs = model3(images)
# #         _, predicted = torch.max(outputs.data, 1)
# #         total += labels.size(0)
# #         correct += (predicted == labels).sum().item()

# #     print('Accuracy of the network on the 10000 test images model 3: {} %'.format(100 * correct / total))

# # Save the model checkpoint
# # torch.save(model.state_dict(), 'model.ckpt')

# Fungsi untuk Training dan Evaluasi Model
def train_and_evaluate(model, train_loader, val_loader, criterion, optimizer, num_epochs):
    val_accuracy = []

    for epoch in range(num_epochs):
        model.train()
        for images, labels in train_loader:
            images = images.reshape(-1, 28 * 28).to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Evaluate on validation set
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.reshape(-1, 28 * 28).to(device)
                labels = labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_acc = 100 * correct / total
        val_accuracy.append(val_acc)

    return val_accuracy

# Definisikan Ruang Hyperparameter
param_grid = {
    'learning_rate': [0.01, 0.001],
    'batch_size': [32, 64],
    'num_epochs': [10, 20]
}

# List untuk menyimpan hasil Grid Search
results = []

# Loop untuk mencoba semua kombinasi hyperparameter
for lr in param_grid['learning_rate']:
    for batch_size in param_grid['batch_size']:
        for num_epochs in param_grid['num_epochs']:
            # Setup DataLoader dengan batch size yang dipilih
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

            # Inisialisasi model, loss function, dan optimizer
            model = NeuralNet(input_size=784, hidden_size1=128, hidden_size2=64, num_classes=10).to(device)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=lr)

            # Train and evaluate model
            val_accuracy = train_and_evaluate(model, train_loader, val_loader, criterion, optimizer, num_epochs)

            # Simpan hasil
            result = {
                'learning_rate': lr,
                'batch_size': batch_size,
                'num_epochs': num_epochs,
                'val_accuracy': val_accuracy
            }
            results.append(result)

            # Print hasil sementara
            print(f"LR: {lr}, Batch Size: {batch_size}, Epochs: {num_epochs} => Final Validation Accuracy: {val_accuracy[-1]:.2f}%")

# Visualisasi Akurasi dari Semua Kombinasi Hyperparameters
plt.figure(figsize=(12, 8))

# Loop melalui semua hasil dan plot akurasi
for result in results:
    plt.plot(result['val_accuracy'], label=f"LR: {result['learning_rate']}, Batch: {result['batch_size']}, Epochs: {result['num_epochs']}")

plt.xlabel('Epoch')
plt.ylabel('Validation Accuracy (%)')
plt.title('Validation Accuracy vs Epochs for Different Hyperparameter Combinations')
plt.legend(loc='best')
plt.grid(True)
plt.show()