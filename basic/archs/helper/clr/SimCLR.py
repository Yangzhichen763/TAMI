import torch
import torch.nn as nn
import torch.nn.functional as F

from basic.archs.helper.clr.basic import MLPHead


class InfoNCELoss(nn.Module):
    """
    InfoNCE Loss (SimCLR-style)
    ---------------------------
    Args:
        batch_size (int): number of original samples before augmentation
        n_views (int): number of views/augmentations per sample, usually 2
        temperature (float): scaling factor for logits (default: 0.07)
        device (torch.device | str | None): device for computation

    Usage:
        >>> loss_fn = InfoNCE(batch_size=256, n_views=2, temperature=0.1)
        >>> loss = loss_fn(features)   # features: [batch_size*n_views, dim]
    """

    def __init__(
            self, n_views: int = 2, temperature: float = 0.07, normalize: bool = True,
            device=None
    ):
        super().__init__()
        self.n_views = n_views
        self.temperature = temperature
        self.normalize = normalize
        self.device = device

    def _init_masks(self, batch_size):
        # register the positive pair label pattern once for efficiency
        labels = torch.cat([torch.arange(batch_size) for _ in range(self.n_views)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        self.register_buffer("labels_mask", labels)

        # also register identity mask for later use
        mask = torch.eye(labels.shape[0], dtype=torch.bool)
        self.register_buffer("diag_mask", mask)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Compute InfoNCE loss given multi-view features.

        Args:
            features: [batch_size * n_views, dim] tensor of embeddings
        Returns:
            loss: scalar tensor
        """
        device = self.device or features.device
        if not hasattr(self, "labels_mask"):
            self._init_masks(features.shape[0] // self.n_views)

        # Normalize features
        if self.normalize:
            features = F.normalize(features, dim=1)

        # Compute cosine similarity matrix
        similarity_matrix = torch.matmul(features, features.T)

        # Move registered buffers to correct device (for safety)
        labels = self.labels_mask.to(device)
        mask = self.diag_mask.to(device)

        # Remove diagonal (self-similarity)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

        # Split into positive and negative pairs
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        # Combine logits: positives first, negatives next
        logits = torch.cat([positives, negatives], dim=1)
        targets = torch.zeros(logits.shape[0], dtype=torch.long, device=device)

        # Scale by temperature
        logits = logits / self.temperature

        # Compute final cross-entropy InfoNCE loss
        loss = F.cross_entropy(logits, targets)
        return loss


class SimCLR(nn.Module):
    def __init__(self, encoder, out_dim=128, hidden_dim=2048, temperature=0.1):
        super().__init__()
        # encoders
        self.encoder = encoder
        self.projector = MLPHead(encoder.output_dim, hidden_dim, out_dim)

        # loss function
        self.loss_fn = InfoNCELoss(temperature=temperature, normalize=True)

    def forward(self, x1, x2):
        # encode two augmented views
        z1 = self.projector(self.encoder(x1))
        z2 = self.projector(self.encoder(x2))

        # compute NT-Xent loss
        loss = self.loss_fn(z1, z2)
        return loss


if __name__ == "__main__":
    import torchvision
    from torchvision import transforms
    from tqdm import tqdm


    class DummyNet(nn.Module):
        def __init__(self, output_dim=512):
            super().__init__()
            self.output_dim = output_dim
            self.conv1 = nn.Conv2d(1, 32, 3, 1)
            self.conv2 = nn.Conv2d(32, 64, 3, 1)
            self.fc1 = nn.Linear(9216, 128)
            self.fc2 = nn.Linear(128, output_dim)

        def forward(self, x):
            x = F.relu(self.conv1(x))
            x = F.max_pool2d(x, 2)
            x = F.relu(self.conv2(x))
            x = F.max_pool2d(x, 2)
            x = torch.flatten(x, 1)
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
            return x


    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_set = torchvision.datasets.MNIST(
        root='./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=256, shuffle=True, num_workers=4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = DummyNet(output_dim=512)
    model = SimCLR(encoder, out_dim=256, hidden_dim=1024).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)


    epochs = 10
    for epoch in range(epochs):
        progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")
        total_loss = 0

        for batch, (x, _) in enumerate(progress):
            x1, x2 = x.to(device), x.to(device) # 上下文视图和目标视图

            loss = model(x1, x2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            model.update()

            total_loss += loss.item()
            avg_loss = total_loss / (batch + 1)
            progress.set_postfix(loss=f"{avg_loss:.4f}")
