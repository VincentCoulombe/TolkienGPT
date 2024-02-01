import torch
import matplotlib.pyplot as plt


class TrainingScheduler(object):
    """Gère le warmup et le scheduler de learning rate"""

    def __init__(
        self,
        optimizer,
        lr0,
        scheduler,
        warmup_iteration=0,
    ):
        self.optimizer = optimizer
        self.warmup_iteration = warmup_iteration
        self.lr_scheduler = scheduler
        self.lr0 = lr0

        if self.warmup_iteration > 0:
            self.step(0.5)  # Set un lr de ((lr0 * 0.5) / warmup_iteration) à l'epoch 0

    def warmup(self, cur_iteration):
        warmup_lr = self.lr0 * float(cur_iteration) / float(self.warmup_iteration)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = warmup_lr

    def step(self, cur_iteration, *args, **kwargs):
        if cur_iteration < self.warmup_iteration:
            self.warmup(cur_iteration)
        elif self.lr_scheduler is not None:
            if "metrics" in kwargs:
                self.lr_scheduler.step(kwargs["metrics"])
            else:
                self.lr_scheduler.step()

    def load_state_dict(self, state_dict):
        if self.lr_scheduler is not None:
            self.lr_scheduler.load_state_dict(state_dict)


if __name__ == "__main__":
    v = torch.zeros(10)
    lr = 0.01
    total_iter = 100
    warmup_iter = 10

    optim = torch.optim.SGD([v], lr=lr)
    scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, total_iter - warmup_iter
    )
    scheduler = TrainingScheduler(
        optimizer=optim,
        warmup_iteration=warmup_iter,
        lr0=lr,
        scheduler=scheduler_cosine,
    )

    x_iter = [0]
    y_lr = [0.0]

    for iter in range(1, total_iter + 1):
        print("iter: ", iter, " ,lr: ", optim.param_groups[0]["lr"])

        optim.zero_grad()
        optim.step()

        scheduler.step(iter)

        x_iter.append(iter)
        y_lr.append(optim.param_groups[0]["lr"])

    plt.plot(x_iter, y_lr, "b")
    plt.legend(["learning rate"])
    plt.xlabel("iteration")
    plt.ylabel("learning rate")
    plt.show()
