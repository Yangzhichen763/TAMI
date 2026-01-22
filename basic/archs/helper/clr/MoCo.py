import torch
import torch.nn as nn
import torch.nn.functional as F

from basic.archs.helper.ema import EMAModule
from basic.archs.helper.clr.basic import MLPHead


class MoCoLoss(nn.Module):
    """
    Momentum Contrastive (InfoNCE) Loss
    -----------------------------------
    Args:
        temperature (float): scaling factor for logits
    """
    def __init__(self, temperature=0.2):
        super().__init__()
        self.temperature = temperature

    def forward(self, q, k, queue):
        """
        Args:
            q: query features [B, D]
            k: key features   [B, D]
            queue: queue of negative keys [D, K]
        Returns:
            loss: scalar tensor
        """
        q = F.normalize(q, dim=1)
        k = F.normalize(k, dim=1)
        queue = F.normalize(queue, dim=0)

        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [q, queue])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)
        logits /= self.temperature

        # labels: positives are the 0-th
        labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
        loss = F.cross_entropy(logits, labels)
        return loss


class MoCo(nn.Module):
    """
    MoCo v2 implementation (BYOL/DINO style)
    ----------------------------------------
    Args:
        encoder: backbone encoder with attribute `output_dim`
        dim: projection dimension
        K: queue size
        m: momentum for updating key encoder
        T: temperature for InfoNCE
    """
    def __init__(self, encoder, dim=128, K=4096, m=0.999, T=0.2, hidden_dim=2048):
        super().__init__()
        self.K = K
        self.m = m
        self.T = T

        # encoders
        self.encoder_q = nn.Sequential(encoder, MLPHead(encoder.output_dim, hidden_dim, dim))
        self.encoder_k = EMAModule(self.encoder_q)

        # loss function
        self.loss_fn = MoCoLoss(temperature=T)

        # queue
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = F.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def dequeue_and_enqueue(self, keys: torch.Tensor) -> None:
        """
        Update the memory queue (FIFO) with the latest keys.

        Args:
            keys (Tensor): normalized key features of shape [B, D]
        """
        keys = F.normalize(keys, dim=1)
        batch_size = keys.size(0)

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0, (
            f"Queue size {self.K} must be divisible by batch size {batch_size}."
        )

        self.queue[:, ptr:ptr + batch_size] = keys.T

        ptr = (ptr + batch_size) % self.K
        self.queue_ptr[0] = ptr

    def forward(self, x1, x2):
        # query features
        q = self.encoder_q(x1)

        # key features
        with torch.no_grad():
            k = self.encoder_k(x2)

        # compute MoCo InfoNCE loss
        loss = self.loss_fn(q, k, self.queue)

        # update queue
        self.dequeue_and_enqueue(k)
        return loss

    def update(self):
        self.encoder_k.update(self.encoder_q)


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
    model = MoCo(encoder, dim=128, K=4096, m=0.999, T=0.2, hidden_dim=1024).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)


    epochs = 10
    for epoch in range(epochs):
        progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")
        total_loss = 0
        for batch, (x, _) in enumerate(progress):
            x1, x2 = x.to(device), x.to(device)

            loss = model(x1, x2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            model.update()

            total_loss += loss.item()
            avg_loss = total_loss / (batch + 1)
            progress.set_postfix(loss=f"{avg_loss:.4f}")