from utils import *
from network import *
from data_process import *


def show_generation(predrnn, input_variable, target_variable, hidden_sizes, device, save_path_index):
    plt.figure(figsize=(30, 9))
    predrnn_hidden = initial_hiddens(1, hidden_sizes, device)

    if device is not None:
        input_variable = input_variable.to(device)
        target_variable = target_variable.to(device)

    predrnn_output = None
# 前向
    for ei in range(10):
        # show input
        plt.subplot(3, 10, 1 + ei)
        downsampled_input = torch.max_pool2d(input_variable[ei], 2)
        plt.imshow(downsampled_input[0, 0].detach())
        predrnn_output, predrnn_hidden = predrnn(downsampled_input, predrnn_hidden)
# 反向
    for di in range(10):
        # show ground truth
        plt.subplot(3, 10, 11 + di)
        downsampled_target = torch.max_pool2d(target_variable[di], 2)
        plt.imshow(downsampled_target[0, 0].detach())

    for di in range(10):
        # show output
        plt.subplot(3, 10, 21 + di)
        plt.imshow(predrnn_output[0, 0].detach())
        predrnn_output, predrnn_hidden = predrnn(predrnn_output, predrnn_hidden)

    print("save image  ---")
    path = r"F:\PYcode\finalAssignment\picture\{}{}.png".format("generated_images_", save_path_index)
    plt.savefig(path)


def show(predrnn, show_data_generator, hidden_sizes, show_data_size, seq_size, image_size, device=None):
    import random

    shows = random.sample(range(0, 25), show_data_size)  # 从25个文件包（一共取了25个用于测试）中随机抽show_data_size个包
    print("保存输出的图片数（张）：", len(shows))

    for show_idx in range(show_data_size):

        idx = shows[show_idx]
        data = show_data_generator[idx: idx + 1]

        show_pair = data_generator(data, 1, seq_size, image_size)  # 返回input_data和target_data

        input_variable = torch.from_numpy(show_pair[0])  # [20, 10, 1, 64, 64] 转化成张量，且与数组共享内存
        target_variable = torch.from_numpy(show_pair[1])

        show_generation(predrnn, input_variable, target_variable, hidden_sizes, device, show_idx)  # 保存图片


def main():
    seq_size = 40
    image_size = 64
    input_size = (1, 32, 32)
    hidden_sizes = [(128, 32, 32), (64, 32, 32),
                    (64, 32, 32), (64, 32, 32)]
    kernel_HWs = [(5, 5)] * 4
    predrnn = PredRNN(input_size, hidden_sizes, kernel_HWs)
    predrnn.load_state_dict(torch.load('F:\\PYcode\\finalAssignment\\models\\model.pkl', map_location='cpu'))

    show_data_generator = get_batch_paths(get_all_parent_path(r'F:\PYcode\finalAssignment\testDataSet'), seq_size)

    # show_data_size = len(show_data_generator)

    show(predrnn, show_data_generator, hidden_sizes, 10, seq_size, image_size)

if __name__ == '__main__':
    main()
