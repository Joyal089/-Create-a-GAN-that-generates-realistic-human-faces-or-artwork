import torch
import matplotlib.pyplot as plt
from torchvision.utils import save_image

# Define the Generator Network
class Generator(torch.nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(100, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, 64 * 64 * 3),  # Output shape: 64x64 RGB images
            torch.nn.Tanh()
        )

    def forward(self, z):
        return self.model(z).view(z.size(0), 3, 64, 64)  # Reshape to image size

# Load the pre-trained generator
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
generator = Generator().to(device)

# Load saved model weights
generator.load_state_dict(torch.load('generator.pth'))
generator.eval()  # Set the generator to evaluation mode
print("Generator model loaded and ready for image generation.")

# Step 1: Generate latent vectors
num_images = 16  # Number of images to generate
z = torch.randn(num_images, 100).to(device)  # Random latent vectors

# Step 2: Generate images
fake_images = generator(z)

# Step 3: Post-process the images
fake_images = (fake_images + 1) / 2  # Rescale images to [0, 1]

# Step 4: Save the generated images
output_file = 'generated_images.png'
save_image(fake_images, output_file, nrow=4)  # Save the images as a single grid
print(f"Generated images saved as {output_file}.")

# Step 5: Visualize the images
fig, axs = plt.subplots(4, 4, figsize=(8, 8))  # Create a 4x4 grid of images
fake_images_np = fake_images.cpu().detach().numpy()  # Convert to NumPy for plotting

for i in range(4):
    for j in range(4):
        axs[i, j].imshow(fake_images_np[i * 4 + j].transpose(1, 2, 0))  # Rearrange dimensions (C, H, W -> H, W, C)
        axs[i, j].axis('off')
plt.show()



import torch
from torchvision.utils import save_image

# Assuming fake_images is already a PyTorch tensor
fake_images = fake_images.float()  # Ensure it's float32

# Rescale if necessary (GAN outputs are usually in range [-1, 1])
fake_images = (fake_images + 1) / 2  # Scale to [0, 1] for image saving

# Save the images
save_image(fake_images, 'generated_images.png', nrow=4)
