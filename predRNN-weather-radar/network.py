from abc import ABC
from ST_LSTM import *


class PredRNN(nn.Module, ABC):
    """
    n_layers is number of layers
    channels is number of channels of each layer. channels[0] is input channels, channel[n_layers] is output channels
    input_HW = (H, W), is height and width of input images
    kernel_HW is height and width of kernels used in convolutions
    """

    def __init__(self, input_size, hidden_sizes, kernel_HWs, kernel_HW_highway=None, kernel_HW_m=None,
                 kernel_HW_output=None, predrnn_only=True):
        super(PredRNN, self).__init__()
        self.n_layers = len(hidden_sizes)  # at least 2
        self.input_size = input_size  # 3-tuple
        self.hidden_sizes = hidden_sizes  # list of 3-tuples of size self.n_layers
        self.kernel_HWs = kernel_HWs
        self.kernel_HW_highway = kernel_HWs[0] if kernel_HW_highway is None else kernel_HW_highway
        self.kernel_HW_m = kernel_HWs[0] if kernel_HW_m is None else kernel_HW_m  # (5, 5)
        self.kernel_HW_output = kernel_HW_output if kernel_HW_output is not None else kernel_HWs[-1]  # (5, 5)

        self.predrnn_only = predrnn_only

        self.cells = nn.ModuleList([])
        for i in range(self.n_layers):
            if i == 0:
                cell = SpatioTemporal_LSTM(self.input_size, self.hidden_sizes[i], self.kernel_HWs[i])
            else:
                cell = SpatioTemporal_LSTM(self.hidden_sizes[i - 1], self.hidden_sizes[i], self.kernel_HWs[i])

            self.cells.append(cell)

        self.highway_unit = HighwayUnit(self.hidden_sizes[0], self.hidden_sizes[0], self.kernel_HW_highway)

        m_padding = (int((self.input_size[1] - self.hidden_sizes[-1][1] + self.kernel_HW_m[0] - 1) / 2),
                     int((self.input_size[2] - self.hidden_sizes[-1][2] + self.kernel_HW_m[1] - 1) / 2))  # (2, 2)

        self.resize_m_conv = nn.Conv2d(self.hidden_sizes[-1][0], self.input_size[0],
                                       self.kernel_HW_m, padding=m_padding)

        if predrnn_only:
            out_padding = (int((self.input_size[1] - self.hidden_sizes[-1][1] + self.kernel_HWs[-1][0] - 1) / 2),
                           int((self.input_size[2] - self.hidden_sizes[-1][2] + self.kernel_HWs[-1][1] - 1) / 2))  # (2, 2)

            self.output_conv = nn.Conv2d(self.hidden_sizes[-1][0], self.input_size[0],
                                         self.kernel_HW_output, padding=out_padding)

        self._reset_parameters()

    def _reset_parameters(self):
        for name, param in self.named_parameters():
            if len(param.size()) == 1:
                if name.find("f.") >= 0:
                    nn.init.constant_(param, 1)
                elif name.find("f_.") >= 0:
                    nn.init.constant_(param, 1)
                else:
                    nn.init.constant_(param, 0)
            elif len(param.size()) == 4:
                ran = 0.09 / (param.size()[1] * param.size()[2] * param.size()[3]) ** 0.3
                nn.init.uniform_(param, -ran, ran)
            else:
                raise Exception('unexpected case')

    def forward(self, input_, states=None):
        assert states is not None, "missing states"
        h, c, m, z = states

        for j, cell in enumerate(self.cells):  # j is layer
            if j == 0:
                # m[-1]         [10, 64, 32, 32]        
                resized_m = self.resize_m_conv(m[-1])   # [10, 1, 32, 32]
                h[j], c[j], m[j] = cell(input_, (h[j], c[j], resized_m))
            elif j == 1:
                z = self.highway_unit(h[0], z)  # [10, 128, 32, 32]
                h[j], c[j], m[j] = cell(z, (h[j], c[j], m[j - 1]))
            else:
                h[j], c[j], m[j] = cell(h[j - 1], (h[j], c[j], m[j - 1]))
        if not self.predrnn_only:
            output = h[-1]
        else:
            output = self.output_conv(h[-1])    # [10, 1, 32, 32]
        return output, (h, c, m, z)
