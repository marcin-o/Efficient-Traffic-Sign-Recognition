import torch
import torch.nn as nn
import torch.nn.functional as F


class DistillationLoss(nn.Module):
    def __init__(self, temperature=4.0, alpha=0.7):
        super().__init__()
        self.T = temperature
        self.alpha = alpha
        self.ce = nn.CrossEntropyLoss()

    def forward(self, student_logits, teacher_logits, labels):
        soft_loss = F.kl_div(
            F.log_softmax(student_logits / self.T, dim=1),
            F.softmax(teacher_logits / self.T, dim=1),
            reduction="batchmean"
        ) * (self.T ** 2)

        hard_loss = self.ce(student_logits, labels)
        return self.alpha * soft_loss + (1 - self.alpha) * hard_loss


def distill_train(student, teacher, loader, optimizer, criterion, device, scaler=None):
    student.train()
    teacher.eval()
    total_loss = 0.0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()

        with torch.no_grad():
            t_logits = teacher(x)

        if scaler is not None:
            with torch.cuda.amp.autocast():
                s_logits = student(x)
                loss = criterion(s_logits, t_logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            s_logits = student(x)
            loss = criterion(s_logits, t_logits, y)
            loss.backward()
            optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)