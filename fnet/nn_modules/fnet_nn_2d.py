import torch
import pdb

class Net(torch.nn.Module):
    def __init__(self,
                dropout_p=0.2,
                in_channels=1,
                out_channels=1,
                depth = 4):
        super().__init__()
        mult_chan=32
        
        
        self.net_recurse = _Net_recurse(n_in_channels=in_channels, mult_chan=mult_chan, depth=depth, dropout_p=dropout_p)
        self.conv_out = torch.nn.Conv2d(mult_chan*in_channels,  out_channels, kernel_size=3, padding=1) #mult_chan*2 for 2 channels

    def forward(self, x):
        x_rec = self.net_recurse(x)
        return self.conv_out(x_rec)

class _Net_recurse(torch.nn.Module):
    def __init__(self, n_in_channels, mult_chan=2, depth=0, dropout_p=0.2):
        """Class for recursive definition of U-network.p

        Parameters:
        in_channels - (int) number of channels for input.
        mult_chan - (int) factor to determine number of output channels
        depth - (int) if 0, this subnet will only be convolutions that double the channel count.
        """
        super().__init__()
        self.depth = depth
        n_out_channels = n_in_channels*mult_chan
        self.sub_2conv_more = SubNet2Conv(n_in_channels, n_out_channels, dropout_p)
        
        if depth > 0:
            self.sub_2conv_less = SubNet2Conv(2*n_out_channels, n_out_channels, dropout_p)
            self.conv_down = torch.nn.Conv2d(n_out_channels, n_out_channels, 2, stride=2)
            self.bn0 = torch.nn.BatchNorm2d(n_out_channels)
            self.relu0 = torch.nn.ReLU()
            self.dropout0 = torch.nn.Dropout(dropout_p)
            
            self.convt = torch.nn.ConvTranspose2d(2*n_out_channels, n_out_channels, kernel_size=2, stride=2)
            self.bn1 = torch.nn.BatchNorm2d(n_out_channels)
            self.relu1 = torch.nn.ReLU()
            self.dropout1 = torch.nn.Dropout(dropout_p)
            self.sub_u = _Net_recurse(n_out_channels, mult_chan=2, depth=(depth - 1))
            
    def forward(self, x):
        if self.depth == 0:
            return self.sub_2conv_more(x)
        else:  # depth > 0
            x_2conv_more = self.sub_2conv_more(x)
            x_conv_down = self.conv_down(x_2conv_more)
            x_bn0 = self.bn0(x_conv_down)
            x_relu0 = self.relu0(x_bn0)
            x_drop0 = self.dropout0(x_relu0)
            x_sub_u = self.sub_u(x_drop0) #x_relu0)
            x_convt = self.convt(x_sub_u)
            x_bn1 = self.bn1(x_convt)
            x_relu1 = self.relu1(x_bn1)
            x_drop1 = self.dropout1(x_relu1)
            x_cat = torch.cat((x_2conv_more, x_relu1), 1)  # concatenate
            x_2conv_less = self.sub_2conv_less(x_cat)
        return x_2conv_less

class SubNet2Conv(torch.nn.Module):
    def __init__(self, n_in, n_out, dropout_p=0.2):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(n_in,  n_out, kernel_size=3, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(n_out)
        self.relu1 = torch.nn.ReLU()
        self.dropout1 = torch.nn.Dropout(dropout_p)
        self.conv2 = torch.nn.Conv2d(n_out, n_out, kernel_size=3, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(n_out)
        self.relu2 = torch.nn.ReLU()
        self.dropout2 = torch.nn.Dropout(dropout_p)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        return x

