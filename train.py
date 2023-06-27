import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torch.utils.tensorboard import SummaryWriter
from model import Discriminator, Generator, initialize_weights
# used for create parameters
import argparse
from dataset import Downloader, CelebADataset
from utils import gradient_penalty

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Running on {device}')

LEARNING_RATE = 1e-4
BATCH_SIZE = 64
IMAGE_SIZE = 64
Z_DIM = 100
NUM_EPOCHS = 100
FEATURES_C = 16
FEATURES_G = 16
C_ITERATIONS = 5
LAMBDA_GP = 10
CHANNELS_IMG = 1
NUM_CLASSES = 10
G_EMBEDDING = 100

# print(CHANNELS_IMG)

transforms = transforms.Compose(
    [
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.5 for _ in range(CHANNELS_IMG)], [0.5 for _ in range(CHANNELS_IMG)]),
    ]
)

dataset = datasets.MNIST(root="dataset/", train=True, transform=transforms, download=True)

# use drop_last=True to drop the last batch if it is not equal to BATCH_SIZE
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True ,drop_last=True)

# initialize generator and discriminator
# discriminator should be called critic
G = Generator(Z_DIM, CHANNELS_IMG, FEATURES_G, NUM_CLASSES, IMAGE_SIZE, G_EMBEDDING).to(device)
C = Discriminator(CHANNELS_IMG, FEATURES_C, NUM_CLASSES, IMAGE_SIZE).to(device)
print(G)
print(C)
initialize_weights(G)
initialize_weights(C)

# initialize optimizer
optim_G = optim.Adam(G.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.9))
optim_C = optim.Adam(C.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.9))


fixed_noise = torch.randn(32, Z_DIM, 1, 1).to(device)
writer_real = SummaryWriter(f"logs/GAN_MNIST/real")
writer_fake = SummaryWriter(f"logs/GAN_MNIST/fake")
step = 0

G.train()
C.train()

for epoch in range(NUM_EPOCHS):
    for batch_idx, (real, labels) in enumerate(dataloader):
        real = real.to(device)
        cur_batch_size = real.shape[0]
        labels = labels.to(device)

        # Train Critic: max E[critic(real)] - E[critic(fake)]
        for _ in range(C_ITERATIONS):
            noise = torch.randn((BATCH_SIZE, Z_DIM, 1, 1)).to(device)
            fake = G(noise, labels)
            C_real = C(real, labels).reshape(-1)
            C_fake = C(fake, labels).reshape(-1)
            gp = gradient_penalty(C, labels, real, fake, device=device)
            loss_C = -(torch.mean(C_real) - torch.mean(C_fake)) + LAMBDA_GP * gp
            C.zero_grad()
            loss_C.backward(retain_graph=True)
            optim_C.step()

        # Train Generator: max E[critic(gen_fake)] <-> min -E[critic(gen_fake)]
        G_fake = C(fake, labels).reshape(-1)
        loss_G = -torch.mean(G_fake)
        G.zero_grad()
        loss_G.backward()
        optim_G.step()

        if batch_idx % 5 == 0 and batch_idx > 0:
            print(
                f"\rEpoch [{epoch}/{NUM_EPOCHS}] Batch {batch_idx}/{len(dataloader)} \
                  Loss D: {loss_C:.4f}, loss G: {loss_G:.4f}",
                end=""
            )
        if batch_idx % 100 == 0:
            with torch.no_grad():
                fake = G(noise, labels)
                # take out (up to) 32 examples
                img_grid_real = torchvision.utils.make_grid(
                    real[:32], normalize=True
                )
                img_grid_fake = torchvision.utils.make_grid(
                    fake[:32], normalize=True
                )

                writer_real.add_image(
                    "Real Images", img_grid_real, global_step=step
                )
                writer_fake.add_image(
                    "Fake Images", img_grid_fake, global_step=step
                )
            step += 1