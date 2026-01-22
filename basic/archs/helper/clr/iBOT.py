import torch
import torch.nn as nn
import torch.nn.functional as F

from basic.archs.helper.ema import EMAModule


class iBOTLoss(nn.Module):
    def __init__(
            self, out_dim, patch_out_dim,
            teacher_temp=0.04, teacher_patch_temp=0.07,
            student_temp=0.1, lambda1=1.0, lambda2=1.0,
            center_momentum=0.9
    ):
        super().__init__()
        self.student_temp = student_temp
        self.teacher_temp = teacher_temp
        self.teacher_patch_temp = teacher_patch_temp
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.center_momentum = center_momentum

        self.register_buffer("center", torch.zeros(1, out_dim))
        self.register_buffer("center_patch", torch.zeros(1, 1, patch_out_dim))

    def forward(self, student_out, teacher_out):
        s_cls, s_patch = student_out
        t_cls, t_patch = teacher_out

        # student: log_softmax
        s_cls = F.log_softmax(s_cls / self.student_temp, dim=-1)
        s_patch = F.log_softmax(s_patch / self.student_temp, dim=-1)

        # teacher: softmax after centering & sharpening
        t_cls = F.softmax((t_cls - self.center) / self.teacher_temp, dim=-1).detach()
        t_patch = F.softmax((t_patch - self.center_patch) / self.teacher_patch_temp, dim=-1).detach()

        # iBOT loss: Symmetric KL divergence
        # loss_cls = -(t_cls * s_cls).sum(dim=-1).mean()
        # loss_patch = -(t_patch * s_patch).sum(dim=-1).mean()
        loss_cls = F.kl_div(s_cls, t_cls, reduction='batchmean')
        loss_patch = F.kl_div(s_patch, t_patch, reduction='batchmean')
        loss = self.lambda1 * loss_cls + self.lambda2 * loss_patch

        # update center
        self.update_center(t_cls, t_patch)
        return loss

    @torch.no_grad()
    def update_center(self, teacher_cls, teacher_patch):
        batch_center = torch.sum(teacher_cls, dim=0, keepdim=True) / len(teacher_cls)
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)

        patch_center = torch.sum(teacher_patch.mean(1), dim=0, keepdim=True) / len(teacher_patch)
        self.center_patch = self.center2 * self.center_momentum + patch_center * (1 - self.center_momentum)


class iBOT(nn.Module):
    def __init__(self, encoder, out_dim=8192, patch_out_dim=8192, ema_decay=0.996):
        super().__init__()
        # encoders
        self.student = encoder
        self.teacher = EMAModule(encoder, ema_decay)

        # loss function
        self.loss_fn = iBOTLoss(out_dim, patch_out_dim)

    def forward(self, x):
        # student
        s_out = self.student(x)

        # teacher
        with torch.no_grad():
            t_out = self.teacher(x)

        # iBOT loss: Symmetric KL divergence
        loss = self.loss_fn(s_out, t_out)
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
    model = iBOT(encoder, out_dim=256, patch_out_dim=256).to(device)
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
