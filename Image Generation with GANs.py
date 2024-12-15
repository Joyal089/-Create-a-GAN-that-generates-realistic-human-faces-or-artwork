# Create a GAN that generates realistic human faces or artwork (e.g., using a dataset like CelebA or art images).
# Project Idea: Build a Generative Adversarial Network (GAN) to generate realistic images. You can use datasets like MNIST (for handwritten digits) or CelebA (for celebrity faces).
    # Skills Used: GANs, deep learning, image processing.
    # Tools/Frameworks: TensorFlow or PyTorch.
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Define the Generator Network
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(100, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 64 * 64 * 3),  # Output shape: 64x64 RGB images
            nn.Tanh()
        )

    def forward(self, z):
        return self.model(z).view(z.size(0), 3, 64, 64)  # Reshape to image size

# Define the Discriminator Network
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(64 * 64 * 3, 1024),  # Flatten input for fully connected layers
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x.view(x.size(0), -1))  # Flatten the image

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize the models
generator = Generator().to(device)
discriminator = Discriminator().to(device)

# Loss function
criterion = nn.BCELoss()

# Optimizers
lr = 0.0002
betas = (0.5, 0.999)
optimizer_g = optim.Adam(generator.parameters(), lr=lr, betas=betas)
optimizer_d = optim.Adam(discriminator.parameters(), lr=lr, betas=betas)

# DataLoader (using CIFAR-10 dataset for example)
transform = transforms.Compose([
    transforms.Resize(64),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

dataset = datasets.CIFAR10(root='./data', download=True, train=True, transform=transform)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# Check if saved models exist
train_model = not (os.path.exists('generator.pth') and os.path.exists('discriminator.pth'))

if not train_model:
    print("Pre-trained models found. Loading models...")
    generator.load_state_dict(torch.load('generator.pth'))
    discriminator.load_state_dict(torch.load('discriminator.pth'))
    generator.eval()
    discriminator.eval()
else:
    print("No pre-trained models found. Starting training...")

    # Training Loop
    num_epochs = 50
    for epoch in range(num_epochs):
        for i, (imgs, _) in enumerate(dataloader):
            # Move data to device
            imgs = imgs.to(device)

            # Create labels
            real_labels = torch.ones(imgs.size(0), 1).to(device)
            fake_labels = torch.zeros(imgs.size(0), 1).to(device)

            # Train Discriminator
            optimizer_d.zero_grad()

            # Real images
            real_preds = discriminator(imgs)
            d_loss_real = criterion(real_preds, real_labels)
            d_loss_real.backward()

            # Fake images
            z = torch.randn(imgs.size(0), 100).to(device)
            fake_imgs = generator(z)
            fake_preds = discriminator(fake_imgs.detach())
            d_loss_fake = criterion(fake_preds, fake_labels)
            d_loss_fake.backward()

            # Total discriminator loss
            d_loss = d_loss_real + d_loss_fake
            optimizer_d.step()

            # Train Generator
            optimizer_g.zero_grad()

            # Fake images
            fake_preds = discriminator(fake_imgs)
            g_loss = criterion(fake_preds, real_labels)
            g_loss.backward()

            optimizer_g.step()

            # Print stats
            if i % 100 == 0:
                print(f'Epoch [{epoch}/{num_epochs}], Step [{i}/{len(dataloader)}], '
                      f'D Loss: {d_loss.item()}, G Loss: {g_loss.item()}')

        # Save generated images every few epochs or after training
        if (epoch + 1) % 10 == 0:
            torch.save(generator.state_dict(), f'generator_epoch_{epoch+1}.pth')
            torch.save(discriminator.state_dict(), f'discriminator_epoch_{epoch+1}.pth')

    print("Training complete!")

    # Save the models
    torch.save(generator.state_dict(), 'generator.pth')
    torch.save(discriminator.state_dict(), 'discriminator.pth')
    print("Models saved successfully!")

# Evaluate the generator or use it for image generation
print("Ready for image generation or further tasks!")

z = torch.randn(16, 100).to(device)  # Generate a batch of 16 random latent vectors
fake_images = generator(z)
fake_images = (fake_images + 1) / 2  # Rescale to [0, 1]
import matplotlib.pyplot as plt

# Convert images to CPU and NumPy format for visualization
fake_images = fake_images.cpu().detach().numpy()

# Plot some of the generated images
fig, axs = plt.subplots(4, 4, figsize=(8, 8))  # 4x4 grid
for i in range(4):
    for j in range(4):
        axs[i, j].imshow(fake_images[i * 4 + j].transpose(1, 2, 0))  # Rearrange dimensions (C, H, W -> H, W, C)
        axs[i, j].axis('off')
plt.show()
from torchvision.utils import save_image

# Save the first 16 images as a single grid
save_image(fake_images, 'generated_images.png', nrow=4)

z = torch.randn(16, 100).to(device)
new_fake_images = generator(z)

