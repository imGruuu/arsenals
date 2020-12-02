from utils import *
from network import *
from data_process import *


def evaluate(predrnn, eval_data_generator, hidden_sizes,
             eval_data_size, batch_size, seq_size, image_size,
             device):
    loss = []
    batch = 0
    while batch < eval_data_size:
        data = eval_data_generator[batch: batch + batch_size]  # 一次处理一批
        test_pair = data_generator(data, batch_size, seq_size, image_size)  # test_pair存input_data和target_data
        batch += batch_size

        input_variable = torch.from_numpy(test_pair[0])  # [10, 10, 1, 64, 64] 转化成张量，且与数组共享内存
        target_variable = torch.from_numpy(test_pair[1])

        input_length = input_variable.size()[0]
        target_length = target_variable.size()[0]

        predrnn_hidden = initial_hiddens(batch_size, hidden_sizes, device)  # 初始化隐藏层维度
        predrnn_output = None

        if device is not None:
            input_variable = input_variable.to(device)  # copy一份到GPU上，并在GPU上进行运算
            target_variable = target_variable.to(device)
# 前向
        for ei in range(input_length):
            downsampled_input = torch.max_pool2d(input_variable[ei], 2)
            predrnn_output, predrnn_hidden = predrnn(downsampled_input, predrnn_hidden)
# 反向
        sigle_loss = 0.0
        loss_criterion = nn.MSELoss(reduction='mean')  # 损失函数
        for di in range(target_length):
            # 下采样缩小图像尺寸，数据维度以及图片大小
            downsampled_target = torch.max_pool2d(target_variable[di], 2)

            # 根据损失函数：计算损失
            sigle_loss += loss_criterion(predrnn_output, downsampled_target)
            predrnn_output, predrnn_hidden = predrnn(predrnn_output, predrnn_hidden)  # 更新权重

        # 打印效果
        print("\n batch =", batch, "     eval loss =", sigle_loss)
        loss.append(sigle_loss.item() / (batch_size * target_length))

    return loss


def main():
    seq_size = 40  # 序列长度
    image_size = 64  # 图片尺寸
    input_size = (1, 32, 32)  # 输入尺寸 单通，32*32
    hidden_sizes = [(128, 32, 32), (64, 32, 32),
                    (64, 32, 32), (64, 32, 32)]
    kernel_HWs = [(5, 5)] * 4  # 卷积核尺寸
    predrnn = PredRNN(input_size, hidden_sizes, kernel_HWs)  # 每个ST-LSTM单元功能
    predrnn.load_state_dict(torch.load('F:\\PYcode\\finalAssignment\\models\\model.pkl', map_location='cpu'))

    eval_data_generator = get_batch_paths(get_all_parent_path(r'F:\PYcode\finalAssignment\testDataSet'), seq_size)

    losses = evaluate(predrnn, eval_data_generator, hidden_sizes, 25, 1, seq_size, image_size, None)

    sum = 0
    for loss in losses:
        print(loss)
        sum += loss
    print("average loss : ", sum/25.0)


if __name__ == '__main__':
    main()
