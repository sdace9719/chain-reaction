import torch
import torch.optim as optim
import torch.optim.lr_scheduler as schedulers
import matplotlib.pyplot as plt

def make_opt_scheduler(name):
    param = torch.nn.Parameter(torch.tensor(1.0))
    lr = 0.1
    if name == 'StepLR':
        opt = optim.SGD([param], lr=lr)
        sched = schedulers.StepLR(opt, step_size=30, gamma=0.1)
    elif name == 'MultiStepLR':
        opt = optim.SGD([param], lr=lr)
        sched = schedulers.MultiStepLR(opt, milestones=[30, 80], gamma=0.1)
    elif name == 'ExponentialLR':
        opt = optim.SGD([param], lr=lr)
        sched = schedulers.ExponentialLR(opt, gamma=0.95)
    elif name == 'CosineAnnealingLR':
        opt = optim.SGD([param], lr=lr)
        sched = schedulers.CosineAnnealingLR(opt, T_max=100)
    elif name == 'CosineAnnealingWarmRestarts':
        opt = optim.SGD([param], lr=lr)
        sched = schedulers.CosineAnnealingWarmRestarts(opt, T_0=30, T_mult=2)
    elif name == 'LambdaLR':
        opt = optim.SGD([param], lr=lr)
        sched = schedulers.LambdaLR(opt, lr_lambda=lambda epoch: 0.95 ** epoch)
    elif name == 'ReduceLROnPlateau':
        opt = optim.SGD([param], lr=lr)
        sched = schedulers.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=10)
    elif name == 'CyclicLR':
        opt = optim.SGD([param], lr=lr)
        sched = schedulers.CyclicLR(opt, base_lr=0.01, max_lr=0.1, step_size_up=50)
    elif name == 'OneCycleLR':
        opt = optim.SGD([param], lr=lr)
        sched = schedulers.OneCycleLR(opt, max_lr=0.1, total_steps=100)
    elif name == 'PolynomialLR':
        opt = optim.SGD([param], lr=lr)
        sched = schedulers.PolynomialLR(opt, total_iters=100, power=0.9)
    elif name == 'ChainedScheduler':
        opt = optim.SGD([param], lr=lr)
        s1 = schedulers.StepLR(opt, step_size=30, gamma=0.1)
        s2 = schedulers.ExponentialLR(opt, gamma=0.95)
        sched = schedulers.ChainedScheduler([s1, s2])
    elif name == 'SequentialLR':
        opt = optim.SGD([param], lr=lr)
        s1 = schedulers.StepLR(opt, step_size=50, gamma=0.1)
        s2 = schedulers.ExponentialLR(opt, gamma=0.5)
        sched = schedulers.SequentialLR(opt, [s1, s2], milestones=[70])
    else:
        return None, None
    return opt, sched

scheduler_names = [
    'StepLR', 'MultiStepLR', 'ExponentialLR', 'CosineAnnealingLR',
    'CosineAnnealingWarmRestarts', 'LambdaLR', 'ReduceLROnPlateau',
    'CyclicLR', 'OneCycleLR', 'PolynomialLR', 'ChainedScheduler', 'SequentialLR'
]

num_epochs = 100
metrics = [1.0 / (epoch + 1) for epoch in range(num_epochs)]

for name in scheduler_names:
    opt, sched = make_opt_scheduler(name)
    if opt is None:
        continue
    lrs = []
    for epoch in range(num_epochs):
        lrs.append(opt.param_groups[0]['lr'])
        if name == 'ReduceLROnPlateau':
            sched.step(metrics[epoch])
        else:
            sched.step()
    plt.figure()
    plt.plot(range(num_epochs), lrs)
    plt.title(name)
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.show()
