from network import *
from data_process import *
from utils import *
import random
import time
import logging

logging.basicConfig(level=logging.ERROR)


def train(input_variable, target_variable, predrnn, predrnn_optimizer,
          hidden_sizes, device, thresh=100, teacher_forcing_ratio=0.5):
    predrnn_optimizer.zero_grad()

    input_length = input_variable.size()[0]
    target_length = target_variable.size()[0]
    batch_size = target_variable.size()[1]

    # 初始化
    predrnn_hidden = initial_hiddens(input_variable.size()[1], hidden_sizes, device)  # 隐藏层维度
    predrnn_output = None

    # 前向传播
    for ei in range(input_length):
        # 10, 1, 
        downsampled_input = F.max_pool2d(input_variable[ei], 2)  # [10, 1, 32, 32]
        predrnn_output, predrnn_hidden = predrnn(downsampled_input, predrnn_hidden)

        logging.debug("input step{}".format(ei))
        logging.debug("h {}\n c {}\n m {}\n z {}".format(*activation_statistics(predrnn_hidden)))

    # 反向传播
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False  # 是否强制学习 （设50%概率）
    loss = 0.0

    for di in range(target_length):  # [10, 10, 1, 64, 64]
        # 下采样缩小图像尺寸，数据维度以及图片大小
        downsampled_target = torch.max_pool2d(target_variable[di], 2)

        # 根据损失函数：计算损失
        loss += train_criterion(predrnn_output, downsampled_target)

        if use_teacher_forcing:  # 进行强制学习
            predrnn_input = downsampled_target
        else:
            predrnn_input = predrnn_output

        predrnn_output, predrnn_hidden = predrnn(predrnn_input, predrnn_hidden)

        logging.debug("output step{}".format(di))
        logging.debug("h {}\n c {}\n m {}\n z {}".format(*activation_statistics(predrnn_hidden)))

    loss = loss / (target_length * batch_size)
    loss.backward()

    # 根据阈值进行归一化
    nn.utils.clip_grad_norm_(predrnn.parameters(), thresh)
    logging.info("predrnn grad statistics {}".format(grad_statistics(predrnn)))

    # 更新权重
    predrnn_optimizer.step()

    return loss.item()


def trainIters(predrnn, train_data_generator, hidden_sizes,
               seq_size, batch_size, image_size, data_set_size, para_dir, device,
               n_iters=2, print_every=10, plot_every=20, save_every=100,
               thresh=100, learning_rate=0.001, teacher_forcing_ratio=0.5):
    # 计时
    start_time = time.time()

    # 模型的优化器
    predrnn_optimizer = torch.optim.Adam(predrnn.parameters(), lr=learning_rate, amsgrad=True)

    print_loss_total = 0.0
    plot_loss_total = 0.0
    plot_losses = []

    # 进行训练
    for it in range(1, n_iters + 1):
        print("\n echo =", it, "     ======\n")
        batch = 0
        while batch < data_set_size:
            print("\n batch =", batch)
            data_batch = train_data_generator[batch: batch + batch_size]
            train_pair = data_generator(data_batch, batch_size, seq_size, image_size)
            batch += batch_size

            input_variable = torch.from_numpy(train_pair[0])  # [10, 10, 1, 64, 64] 转化成张量，且与数组共享内存
            target_variable = torch.from_numpy(train_pair[1])

            if device is not None:
                input_variable = input_variable.to(device)  # copy一份到GPU上，并在GPU上进行运算
                target_variable = target_variable.to(device)

            # 训练返回损失值
            loss = train(input_variable, target_variable, predrnn, predrnn_optimizer,
                         hidden_sizes, device, thresh=thresh, teacher_forcing_ratio=teacher_forcing_ratio)

            print_loss_total += loss
            plot_loss_total += loss

            if it % print_every == 0:  # 每隔几个打印
                print_loss_avg = print_loss_total / print_every
                print_loss_total = 0.0
                print('%s (%d %d%%) %.4f' % (timeSince(start_time, it / n_iters),
                                             it, it / n_iters * 100, print_loss_avg))

            if it % plot_every == 0:  # 每隔几个记录损失
                plot_loss_avg = plot_loss_total / plot_every
                plot_losses.append(plot_loss_avg)
                plot_loss_total = 0.0

            if it % save_every == 0:  # 每隔几个保存更新权重
                torch.save(predrnn.state_dict(), para_dir)
                showPlot(plot_losses)


def main(batch_size: int = 1, seq_size: int = 40, image_size: int = 64):
    para_dir = 'F:\\PYcode\\finalAssignment\\parameters\\model.pkl'

    size = int(image_size / 2)
    input_size = (1, size, size)
    hidden_sizes = [(128, size, size), (64, size, size),
                    (64, size, size), (64, size, size)]
    kernel_HWs = [(5, 5)] * 4

    predrnn = PredRNN(input_size, hidden_sizes, kernel_HWs)

    try:
        predrnn.load_state_dict(torch.load(para_dir))  # 加载数据到网络中
    except FileNotFoundError:
        pass

    device = torch.device("cuda:0")
    print('device =', device)
    predrnn = predrnn.to(device)  # copy一份到GPU，并在GPU上进行运算

    train_data_generator = get_batch_paths(get_all_parent_path('F:\\PYcode\\finalAssignment\\partDataSet'), seq_size)  # 获取数据集
    data_set_size = len(train_data_generator)
    print('data_set_size =', data_set_size)

    trainIters(predrnn, train_data_generator, hidden_sizes,
               seq_size, batch_size, image_size, data_set_size, para_dir,
               device=device, thresh=image_size, learning_rate=0.0001)

    store_dir = 'F:\\PYcode\\finalAssignment\\models\\model.pkl'
    torch.save(predrnn.state_dict(), store_dir)  # 保存权重


if __name__ == '__main__':
    main()
