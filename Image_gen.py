import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable
from torchvision import datasets, transforms
import numpy as np
import os

# Define the Generator model
class Generator(nn.Module):
    def __init__(self, latent_dim, num_classes, image_size):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.image_size = image_size
        
        self.embed = nn.Embedding(num_classes, 50)
        self.fc = nn.Linear(latent_dim + 50, 128 * (image_size // 4) * (image_size // 4))
        self.conv1 = nn.ConvTranspose2d(128, 64, 4, 2, 1)
        self.conv2 = nn.ConvTranspose2d(64, 3, 4, 2, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, noise, labels):
        embedded_labels = self.embed(labels).view(-1, 50)
        concat_input = torch.cat((noise, embedded_labels), -1)
        x = self.fc(concat_input)
        x = self.relu(x)
        x = x.view(-1, 128, (self.image_size // 4), (self.image_size // 4))
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.sigmoid(x)
        return x

# Define the Discriminator model
class Discriminator(nn.Module):
    def __init__(self, image_size, num_classes):
        super(Discriminator, self).__init__()
        self.image_size = image_size
        self.num_classes = num_classes
        
        self.conv1 = nn.Conv2d(3, 64, 4, 2, 1)
        self.conv2 = nn.Conv2d(64, 128, 4, 2, 1)
        self.fc1 = nn.Linear(128 * (image_size // 4) * (image_size // 4) + 50, 1)
        self.fc2 = nn.Linear(num_classes, 50)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, images, labels):
        x = self.conv1(images)
        x = self.leaky_relu(x)
        x = self.conv2(x)
        x = self.leaky_relu(x)
        x = x.view(-1, 128 * (self.image_size // 4) * (self.image_size // 4))
        label_embedding = self.fc2(labels)
        x = torch.cat((x, label_embedding), -1)
        x = self.fc1(x)
        x = self.sigmoid(x)
        return x

# Define the dataset class
class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = os.listdir(root_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.images[idx])
        image = Image.open(img_name)
        if self.transform:
            image = self.transform(image)
        return image

# Define parameters
latent_dim = 100
num_classes = 1000
image_size = 64
batch_size = 32
num_epochs = 10

# Initialize models
generator = Generator(latent_dim, num_classes, image_size)
discriminator = Discriminator(image_size, num_classes)

# Define loss function and optimizers
criterion = nn.BCELoss()
optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

# Load dataset and create dataloader
transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.CenterCrop(image_size),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])
dataset = CustomDataset(root_dir='your_dataset_path', transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Training loop
def train(generator, discriminator, num_epochs, dataloader):
    for epoch in range(num_epochs):
        for i, images in enumerate(dataloader):
            batch_size = images.size(0)
            real_labels = torch.ones(batch_size, 1)
            fake_labels = torch.zeros(batch_size, 1)
            real_images = Variable(images)
            labels = Variable(torch.randint(0, num_classes, (batch_size,)))
            
            # Train discriminator with real images
            discriminator.zero_grad()
            real_outputs = discriminator(real_images, labels)
            d_loss_real = criterion(real_outputs, real_labels)
            d_loss_real.backward()
            
            # Train discriminator with fake images
            noise = Variable(torch.randn(batch_size, latent_dim))
            fake_images = generator(noise, labels)
            fake_outputs = discriminator(fake_images.detach(), labels)
            d_loss_fake = criterion(fake_outputs, fake_labels)
            d_loss_fake.backward()
            optimizer_D.step()
            
            # Train generator
            generator.zero_grad()
            outputs = discriminator(fake_images, labels)
            g_loss = criterion(outputs, real_labels)
            g_loss.backward()
            optimizer_G.step()
            
            if (i+1) % 100 == 0:
                print('Epoch [%d/%d], Step [%d/%d], Discriminator Loss: %.4f, Generator Loss: %.4f' % 
                      (epoch+1, num_epochs, i+1, len(dataloader), d_loss_real.item() + d_loss_fake.item(), g_loss.item()))

# Example usage:
# train(generator, discriminator, num_epochs, dataloader)

# Example of generating images from a textual prompt
def generate_image(generator, prompt):
    noise = Variable(torch.randn(1, latent_dim))
    label = Variable(torch.LongTensor([prompt])) # Assuming prompt is the index of the desired class
    with torch.no_grad():
        generated_image = generator(noise, label)
    # Visualize or save the generated image
    return generated_image

