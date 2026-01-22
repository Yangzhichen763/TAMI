import torch
import torch.nn as nn
import torch.nn.functional as F

from basic.archs.helper.ema import EMAModule


class DINOHead(nn.Module):
    def __init__(self, in_dim, hidden_dim=2048, out_dim=65536, bottleneck_dim=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, bottleneck_dim),
            nn.GELU(),
        )
        self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1.0)

    def forward(self, x):
        x = self.mlp(x)
        x = F.normalize(x, dim=-1)
        x = self.last_layer(x)
        return x


class DINOLoss(nn.Module):
    def __init__(
            self, out_dim,
            teacher_temp=0.04, student_temp=0.1,
            center_momentum=0.9
    ):
        super().__init__()
        self.student_temp = student_temp
        self.teacher_temp = teacher_temp
        self.center_momentum = center_momentum

        self.register_buffer("center", torch.zeros(1, out_dim))

    def forward(self, student_output_1, student_output_2, teacher_output_1, teacher_output_2):
        # DINO loss: Symmetric KL divergence
        loss = (
            self.H(teacher_output_1, student_output_1)
            + self.H(teacher_output_2, student_output_2)
        ) / 2

        # update center
        self.update_center(torch.cat([teacher_output_1, teacher_output_2], dim=0))
        return loss

    def H(self, t, s):
        s = F.log_softmax(s / self.student_temp, dim=-1)
        t = F.softmax((t - self.center) / self.teacher_temp, dim=-1)
        return F.kl_div(s, t, reduction="batchmean")

    @torch.no_grad()
    def update_center(self, teacher_outputs):
        batch_center = teacher_outputs.mean(dim=0, keepdim=True)
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)


class DINO(nn.Module):
    def __init__(self, backbone, out_dim=65536, ema_decay=0.996, teacher_temp=0.04, student_temp=0.1, center_momentum=0.9):
        super().__init__()
        # encoders
        head = DINOHead(backbone.output_dim, out_dim=out_dim)
        self.student = nn.Sequential(backbone, head)
        self.teacher = EMAModule(self.student, ema_decay)

        # loss function
        self.loss_fn = DINOLoss(out_dim, teacher_temp, student_temp, center_momentum)

    def forward(self, x1, x2):
        # student
        s1 = self.student(x1)
        s2 = self.student(x2)

        # teacher
        with torch.no_grad():
            t1 = self.teacher(x1)
            t2 = self.teacher(x2)

        # DINO loss: Symmetric KL divergence
        loss = self.loss_fn(s1, s2, t1, t2)
        return loss

    def update(self):
        self.teacher.update(self.student)


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
    model = DINO(encoder, out_dim=1024).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)


    epochs = 10
    for epoch in range(epochs):
        progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")
        total_loss = 0

        for batch, (x, _) in enumerate(progress):
            x1, x2 = x.to(device), x.to(device) # 生成两个增强视图

            loss = model(x1, x2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            model.update()

            total_loss += loss.item()
            avg_loss = total_loss / (batch + 1)
            progress.set_postfix(loss=f"{avg_loss:.4f}")
