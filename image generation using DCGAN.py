import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os
import torchvision.utils as vutils

# Define the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
batch_size = 64
z_dim = 100  # Latent vector size
lr = 0.0002  # Learning rate for both the generator and discriminator
beta1 = 0.5  # Beta1 hyperparameter for Adam optimizer

# Transform to resize images to 64x64 and normalize them
transform = transforms.Compose([
    transforms.Resize((64, 64)),  # Resize images to 64x64
    transforms.ToTensor(),        # Convert image to tensor
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize images to [-1, 1]
])

# Custom dataset class to load images from a single folder
class CustomImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.img_paths = [os.path.join(root_dir, fname) for fname in os.listdir(root_dir) if fname.endswith(('png', 'jpg', 'jpeg'))]

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img

# Load your dataset (path to your images)
dataset = CustomImageDataset(root_dir=r'C:\Users\joyal\Desktop\to become machine learning enginerr\Generative Ai\Image Generation with GANs\img_celeba', transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Define the Generator model
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(z_dim, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            nn.Tanh()  # Output image with values between [-1, 1]
        )
    
    def forward(self, input):
        return self.main(input)

# Define the Discriminator model
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()  # Output a probability value between [0, 1]
        )

    def forward(self, input):
        return self.main(input)

# Initialize the models
generator = Generator().to(device)
discriminator = Discriminator().to(device)

# Loss function (Binary Cross-Entropy)
criterion = nn.BCELoss()

# Optimizers for both the generator and discriminator
optimizer_g = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, 0.999))
optimizer_d = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, 0.999))

# Fixed noise vector for generating sample images during training
fixed_noise = torch.randn(64, z_dim, 1, 1, device=device)

# Training loop
num_epochs = 25  # Adjust as needed
for epoch in range(num_epochs):
    for i, imgs in enumerate(dataloader):
        # Move images to the device
        imgs = imgs.to(device)

        # Create labels for the real and fake images
        real_labels = torch.ones(batch_size, 1).to(device)
        fake_labels = torch.zeros(batch_size, 1).to(device)

        # -----------------
        # Train the Discriminator
        # -----------------
        optimizer_d.zero_grad()

        # Real images
        real_preds = discriminator(imgs)
        d_loss_real = criterion(real_preds.view(-1, 1), real_labels)
        d_loss_real.backward()

        # Fake images
        z = torch.randn(batch_size, z_dim, 1, 1, device=device)  # Random noise for generator
        fake_imgs = generator(z)
        fake_preds = discriminator(fake_imgs.detach())  # Detach to avoid updating generator
        d_loss_fake = criterion(fake_preds.view(-1, 1), fake_labels)
        d_loss_fake.backward()

        # Update discriminator
        d_loss = d_loss_real + d_loss_fake
        optimizer_d.step()

        # -----------------
        # Train the Generator
        # -----------------
        optimizer_g.zero_grad()

        # Generator's loss (based on discriminator's judgment of fake images)
        output = discriminator(fake_imgs)
        g_loss = criterion(output.view(-1, 1), real_labels)
        g_loss.backward()

        # Update generator
        optimizer_g.step()

        # Print progress
        if i % 50 == 0:
            print(f"Epoch [{epoch}/{num_epochs}], Step [{i}/{len(dataloader)}], "
                  f"D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}")

    # Save generated images after each epoch
    vutils.save_image(fake_imgs.data, f"output/fake_images_epoch_{epoch}.png", normalize=True)

# Save the final model
torch.save(generator.state_dict(), "generator.pth")
torch.save(discriminator.state_dict(), "discriminator.pth")

