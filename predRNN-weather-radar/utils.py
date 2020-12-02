import math
import time
import torch
from functools import reduce
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def showPlot(*curves):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=20)
    ax.yaxis.set_major_locator(loc)
    for curve in curves:
        plt.plot(curve)
    plt.savefig('loss curve.png')
    plt.close()


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / percent
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


def initial_hiddens(batch_size, hidden_sizes, device):
    n_layers = len(hidden_sizes)
    h = []
    c = []
    m = []
    if device is None:
        for i in range(n_layers):
            h.append(torch.zeros([batch_size, *hidden_sizes[i]]))
            c.append(torch.zeros([batch_size, *hidden_sizes[i]]))
            m.append(torch.zeros([batch_size, *hidden_sizes[i]]))
        z = torch.zeros(batch_size, *hidden_sizes[0])
        return h, c, m, z
    else:
        for i in range(n_layers):
            h.append(torch.zeros([batch_size, *hidden_sizes[i]], device=device))
            c.append(torch.zeros([batch_size, *hidden_sizes[i]], device=device))
            m.append(torch.zeros([batch_size, *hidden_sizes[i]], device=device))
        z = torch.zeros(batch_size, *hidden_sizes[0], device=device)
        return h, c, m, z


def activation_statistics(states):
    h, c, m, z = states
    h_avg = []
    c_avg = []
    m_avg = []
    n_layers = len(h)
    for s, s_avg in zip((h, c, m), (h_avg, c_avg, m_avg)):
        for i in range(n_layers):
            s_avg.append(round(torch.sum(s[i]).item() / reduce(lambda x, y: x * y, list(s[i].size())), 4))
    z_avg = round(torch.sum(z).item() / reduce(lambda x, y: x * y, list(z.size())), 4)
    return h_avg, c_avg, m_avg, z_avg


def grad_statistics(model):
    total_grad_value = 0.0
    num_variables = 0
    for param in model.parameters():
        if param.grad is not None:
            total_grad_value += torch.sum(torch.abs(param.grad)).item()
            num_variables += reduce(lambda x, y: x * y, list(param.grad.size()))

    return total_grad_value / num_variables


def L1_loss(inputs, targets):
    loss = torch.nn.L1Loss(reduction='sum')
    return loss(inputs, targets)


def L2_loss(inputs, targets):
    loss = torch.nn.MSELoss(reduction='sum')
    return loss(inputs, targets)


def train_criterion(inputs, targets):
    return L1_loss(inputs, targets) + L2_loss(inputs, targets)
