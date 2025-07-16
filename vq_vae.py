import torch
from torch import nn
import torch.nn.functional as F
from torchview import draw_graph
import torch.optim as optim

class VectorQuantizer(nn.Module):
    def __init__(self, codebook_size=1024, latent_dim=2):
        super().__init__()
        self.embedding = nn.Embedding(codebook_size, latent_dim)
        self.embedding.weight.data.uniform_(-1/codebook_size, 1/codebook_size)

        self.latent_dim = latent_dim
        self.codebook_size = codebook_size

    def forward(self, x, efficient=True):
        
        batch_size = x.shape[0]

        if not efficient:

            # Embed [C x L]
            # Data  [B x L]
            
            # add a batch dimension to the embedding and a codesize dimension to data
            emb = self.embedding.weight.unsqueeze(0).repeat(batch_size, 1, 1) # -----> this is inefficient, because you are repeating it batch size times, and it is big
            x = x.unsqueeze(1)
            
            # So now, Embed [B x C x L], Data [B x 1 x L]
            # We need to compare pair-wise distance with the data and all the codes in Embed
            distances = ((x - emb)**2) # [B x C x L]
            distances = torch.sum(distances, dim=-1) # [B x C], we summed the errors on L 
            
        else:
            
            # (L - C)**2 = (L**2 - 2*C*L + C**2)
            L2 = torch.sum(x ** 2, dim=-1, keepdim=True) # [B x 1]
            
            C2 = torch.sum(self.embedding.weight**2, dim=-1).unsqueeze(0) # [1, C]

            LC = (x @ self.embedding.weight.t()) # [B x C]

            distances = L2 + C2 - 2*LC # [B x C]
            
        closest = torch.argmin(distances, dim=-1)
        
        quantized_latents_idx = torch.zeros(batch_size, self.codebook_size, device=x.device) # B x C
        
        batch_idx = torch.arange(batch_size)
        quantized_latents_idx[batch_idx, closest] = 1 # for each element in the batch, make the closest index 1, else 0.

        quantized_latents = quantized_latents_idx @ self.embedding.weight # select the closest weights in the codebook for each element in the batch, [B x L]

        return quantized_latents

        
class LinearVectorQuantizedVAE(nn.Module):
    def __init__(self, codebook_size=512, latent_dim=2):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(32*32, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, latent_dim),
        )

        self.vq = VectorQuantizer(codebook_size, latent_dim)


        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 32*32),
            nn.Sigmoid()
        )

    def forward_enc(self, x):
        z = self.encoder(x)
        return z
    
    def quantize(self, z):
        
        codes = self.vq(z)

        ### Losses ###
        codebook_loss   = torch.mean((codes - z.detach())**2)
        commitment_loss = torch.mean((codes.detach() - z)**2)

        # Reparameterization trick (for Straight Through Estimation)
        codes = z + (codes - z).detach()

        return codes, codebook_loss, commitment_loss

    def forward_dec(self, x):
        codes, codebook_loss, commitment_loss = self.quantize(x)
        decoded = self.decoder(codes)

        return codes, decoded, codebook_loss, commitment_loss
    
    def forward(self, x):
        # for images
        batch, channels, height, width = x.shape

        x = x.flatten()

        z = self.forward_enc(x)

        quantized_latents, decoded, codebook_loss, commitment_loss = self.forward_dec(z)

        decoded = decoded.reshape(batch, channels, height, width)

        return z, quantized_latents, decoded, codebook_loss, commitment_loss

    def loss(self, x, y):

        encoded, quantized_encoded, decoded, codebook_loss, commitment_loss = self.forward(x)

        reconstruction_loss = torch.mean((y - decoded)**2)
        loss = reconstruction_loss + codebook_loss + 0.25*commitment_loss

        return loss


class ConvolutionalVectorQuantizedVAE(nn.Module):
    def __init__(self, in_channels=1, latent_dim=4, codebook_size=512):
        super().__init__()

        self.bottleneck_dim = latent_dim
        self.in_channels = in_channels
        self.codebook_size = codebook_size

        self.encoder_conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=8, kernel_size=5, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(),

            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),

            nn.Conv2d(in_channels=16, out_channels=self.bottleneck_dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.bottleneck_dim),
            nn.ReLU(),
        )

        self.vq = VectorQuantizer(codebook_size=codebook_size, latent_dim=latent_dim)

        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(in_channels=self.bottleneck_dim, out_channels=16, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),

            nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(),

            nn.ConvTranspose2d(in_channels=8, out_channels=in_channels, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.Sigmoid()
        )


    def forward_enc(self, x):
        z = self.encoder_conv(x)
        return z
    
    def quantize(self, z):
        
        codes = self.vq(z)

        ### Losses ###
        codebook_loss   = torch.mean((codes - z.detach())**2)
        commitment_loss = torch.mean((codes.detach() - z)**2)

        # Reparameterization trick (for Straight Through Estimation)
        codes = z + (codes - z).detach()

        return codes, codebook_loss, commitment_loss

    def forward_dec(self, x):
        batch_size, channels, height, width = x.shape

        x = x.permute(0, 2, 3, 1) # b, h, w, c

        x = torch.flatten(x, start_dim=0, end_dim=-2)

        
        codes, codebook_loss, commitment_loss = self.quantize(x)

        codes = codes.reshape(batch_size, height, width, channels).permute(0, 3, 1, 2)

        decoded = self.decoder_conv(codes)

        return codes, decoded, codebook_loss, commitment_loss
    
    def forward(self, x):
        # for images
        batch, channels, height, width = x.shape

        z = self.forward_enc(x)

        quantized_latents, decoded, codebook_loss, commitment_loss = self.forward_dec(z)

        decoded = decoded.reshape(batch, channels, height, width)

        return z, quantized_latents, decoded, codebook_loss, commitment_loss

    def loss(self, x, y):

        encoded, quantized_encoded, decoded, codebook_loss, commitment_loss = self.forward(x)

        reconstruction_loss = torch.mean((y - decoded)**2)
        loss = reconstruction_loss + codebook_loss + 0.25*commitment_loss

        return loss
     


from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt


transform = transforms.Compose([
    transforms.Resize((32, 32)),  # resize to match your model
    transforms.ToTensor()
])

# Load full MNIST dataset
full_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

# Split into train/val (e.g., 50k train, 10k val)
train_size = 50000
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

# Dataloaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=True)  # small batch for visualization

from tqdm import tqdm

def train_vqvae(model, train_loader, val_loader, epochs=10, lr=1e-3, device='cuda'):
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        for x, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            x = x.to(device)

            loss = model.loss(x, x)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)

        # Validation loss
        model.eval()
        with torch.no_grad():
            total_val_loss = 0
            for x, _ in val_loader:
                x = x.to(device)
                val_loss = model.loss(x, x)
                total_val_loss += val_loss.item()

        avg_val_loss = total_val_loss / len(val_loader)

        print(f"Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")


def show_reconstructions(model, test_loader, device='cuda'):
    model.eval()
    with torch.no_grad():
        x, _ = next(iter(test_loader))
        x = x.to(device)
        _, _, decoded, _, _ = model(x)
        
        x = x.cpu()
        decoded = decoded.cpu()

        # Plot
        plt.figure(figsize=(10, 4))
        for i in range(8):
            # Original
            plt.subplot(2, 8, i + 1)
            plt.imshow(x[i, 0], cmap='gray')
            plt.axis('off')

            # Reconstruction
            plt.subplot(2, 8, i + 9)
            plt.imshow(decoded[i, 0], cmap='gray')
            plt.axis('off')
        plt.suptitle("Top: Original | Bottom: Reconstruction")
        plt.tight_layout()
        plt.show()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ConvolutionalVectorQuantizedVAE()

train_vqvae(model, train_loader, val_loader, epochs=10, lr=1e-3, device=device)
show_reconstructions(model, test_loader, device=device)


'''
print(q.enc.weight.grad) # None
print(q.codebook.grad)
print(q.dec.weight.grad)

'''
#model_graph = draw_graph(vq, input_size=(1, 1))
#model_graph.visual_graph.render('vq_vae_graph', format='png', view=True)