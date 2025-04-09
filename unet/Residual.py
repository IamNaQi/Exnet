import torch
import torch.nn as nn

class ResidualDenseBlock_old(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super(ResidualDenseBlock_old, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels * 2, in_channels, kernel_size=3, groups=in_channels, padding=1, bias=True)
        self.conv3 = nn.Conv2d(in_channels * 3, in_channels, kernel_size=3, groups=in_channels, padding=1, bias=True)
        self.conv4 = nn.Conv2d(in_channels * 4, in_channels, kernel_size=3, groups=in_channels, padding=1, bias=True)
        self.conv5 = nn.Conv2d(in_channels * 5, in_channels, kernel_size=3, groups=in_channels, padding=1, bias=True)

    def forward(self, x):
        initial = x 
        out1 = torch.relu(self.conv1(x))
        out2 = torch.relu(self.conv2(torch.cat([x, out1], 1)))
        out3 = torch.relu(self.conv3(torch.cat([x, out1, out2], 1)))
        out4 = torch.relu(self.conv4(torch.cat([x, out1, out2, out3], 1)))
        out5 = self.conv5(torch.cat([x, out1, out2, out3, out4], 1))
        return out5 * 0.2 + initial
    
class ResidualDenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super(ResidualDenseBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, groups=in_channels, padding=1, bias=True)
        self.conv3 = nn.Conv2d(in_channels, in_channels, kernel_size=3, groups=in_channels, padding=1, bias=True)
        self.conv4 = nn.Conv2d(in_channels, in_channels, kernel_size=3, groups=in_channels, padding=1, bias=True)
        self.conv5 = nn.Conv2d(in_channels, in_channels, kernel_size=3, groups=in_channels, padding=1, bias=True)

    def forward(self, x):
        initial = x 
        out1 = torch.relu(self.conv1(x))
        out2 = torch.relu(self.conv2(x + out1))
        out3 = torch.relu(self.conv3(x + out1 + out2))
        out4 = torch.relu(self.conv4(x + out1 + out2 + out3))
        out5 = self.conv5(x + out1 + out2 + out3 + out4)
        return out5 * 0.2 + initial


# # Example usage
if __name__ == "__main__":
    input_tensor = torch.rand(1, 4, 256, 256)  # Batch size 1, 64 channels, 256x256 image
    rdm = ResidualDenseBlock(in_channels=4, growth_rate=32)
    output = rdm(input_tensor)
    print(output.shape)  # Should be [1, 64, 256, 256]
    param = sum(p.numel() for p in rdm.parameters() if p.requires_grad)
    print (param)
