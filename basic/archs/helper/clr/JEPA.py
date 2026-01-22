import torch
import torch.nn as nn
import torch.nn.functional as F

from basic.archs.helper.ema import EMAModule


class JEPA(nn.Module):
    def __init__(self, encoder, predictor, ema_decay=0.99925):
        super().__init__()
        # encoders
        self.context_encoder = encoder
        self.target_encoder = EMAModule(encoder, decay=ema_decay)
        self.predictor = predictor

    def forward(self, x_context, x_target):
        # encode context (student)
        z_c = self.context_encoder(x_context)
        p_c = self.predictor(z_c)

        # encode target (teacher, no grad)
        with torch.no_grad():
            z_t = self.target_encoder(x_target)

        # JEPA loss: MSE in embedding space
        p_c, z_t = F.normalize(p_c, dim=-1), F.normalize(z_t, dim=-1)
        loss = F.mse_loss(p_c, z_t)
        return loss

    @torch.no_grad()
    def update(self):
        self.target_encoder.update(self.context_encoder)


if __name__ == "__main__":
    import torchvision
    from torchvision import transforms
    from tqdm import tqdm
    from basic.archs.helper.clr.basic import MLPHead


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
    model = JEPA(encoder, MLPHead(512, 1024, 256)).to(device)
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
