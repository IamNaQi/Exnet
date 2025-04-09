from .unet_parts import *
from torch.utils.checkpoint import checkpoint

class Extractor(nn.Module):
    def __init__(self, n_channels, n_classes, length):
        super(Extractor, self).__init__()
        self.length = length
        self.weights = nn.Parameter(torch.randn(length, n_channels, n_classes))
        self.bn = nn.BatchNorm2d(n_channels)  # Correctly handle batch normalization

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        x = x.view(batch_size, channels, -1).permute(0, 2, 1)  # Reshape to (batch_size, height*width, channels)

        out = torch.zeros_like(x)
        for i in range(self.length):
            if i == 0:
                out += x @ self.weights[i]
            else:
                out[:, i:] += x[:, :-i] @ self.weights[i]

        out = out.permute(0, 2, 1).view(batch_size, channels, height, width)  # Reshape back
        out = self.bn(out)  # Apply batch normalization
        return out

def initialize_weights(module):
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)

class MSEA_unet_v3(nn.Module):
    def __init__(self, n_channels, dim, n_classes, length=1, bilinear=False):
        super(MSEA_unet_v3, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.inc = DoubleConv(n_channels, dim)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(dim, dim * 2), nn.Dropout(0.3))  # Add Dropout
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(dim * 2, dim * 4), nn.Dropout(0.3))  # Add Dropout
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(dim * 4, dim * 8), nn.Dropout(0.4))  # Add Dropout
        self.down4 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(dim * 8, dim * 16), nn.Dropout(0.4))  # Add Dropout

        self.extractor = Extractor(dim * 16, dim * 16, length)
        
        self.up1 = nn.ConvTranspose2d(dim * 16, dim * 8, kernel_size=2, stride=2)
        self.att1 = AttentionBlock(dim * 8, dim * 8, dim * 4)
        self.conv5 = DoubleConv(dim * 16, dim * 8)

        self.up2 = nn.ConvTranspose2d(dim * 8, dim * 4, kernel_size=2, stride=2)
        self.att2 = AttentionBlock(dim * 4, dim * 4, dim * 2)
        self.conv6 = DoubleConv(dim * 8, dim * 4)

        self.up3 = nn.ConvTranspose2d(dim * 4, dim * 2, kernel_size=2, stride=2)
        self.att3 = AttentionBlock(dim * 2, dim * 2, dim)
        self.conv7 = DoubleConv(dim * 4, dim * 2)

        self.up4 = nn.ConvTranspose2d(dim * 2, dim, kernel_size=2, stride=2)
        self.att4 = AttentionBlock(dim, dim, dim // 2)
        self.conv8 = DoubleConv(dim * 2, dim)

        self.outc = nn.Conv2d(dim, n_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        x51 = self.extractor(x5)

        x6 = self.up1(x51)
        x4 = self.att1(x6, x4)
        x6 = torch.cat([x4, x6], dim=1)
        x6 = self.conv5(x6)

        x7 = self.up2(x6)
        x3 = self.att2(x7, x3)
        x7 = torch.cat([x3, x7], dim=1)
        x7 = self.conv6(x7)

        x8 = self.up3(x7)
        x2 = self.att3(x8, x2)
        x8 = torch.cat([x2, x8], dim=1)
        x8 = self.conv7(x8)

        x9 = self.up4(x8)
        x1 = self.att4(x9, x1)
        x9 = torch.cat([x1, x9], dim=1)
        x9 = self.conv8(x9)

        logits = self.outc(x9)
        return logits

    def use_checkpointing(self):
        self.inc = checkpoint(self.inc)
        self.down1 = checkpoint(self.down1)
        self.down2 = checkpoint(self.down2)
        self.down3 = checkpoint(self.down3)
        self.down4 = checkpoint(self.down4)
        self.up1 = checkpoint(self.up1)
        self.att1 = checkpoint(self.att1)
        self.conv5 = checkpoint(self.conv5)
        self.up2 = checkpoint(self.up2)
        self.att2 = checkpoint(self.att2)
        self.conv6 = checkpoint(self.conv6)
        self.up3 = checkpoint(self.up3)
        self.att3 = checkpoint(self.att3)
        self.conv7 = checkpoint(self.conv7)
        self.up4 = checkpoint(self.up4)
        self.att4 = checkpoint(self.att4)
        self.conv8 = checkpoint(self.conv8)
        self.outc = checkpoint(self.outc)
import time
import torchprofile
if __name__ == "__main__":
    model = MSEA_unet_v3(n_channels=3, dim=32, n_classes=1)
    model.apply(initialize_weights)

    input_tensor = torch.rand(1, 3, 256, 256)
    output = model(input_tensor)
    print(output.shape)

    flops = torchprofile.profile_macs(model, input_tensor)
    print(f"FLOPs: {flops / 1e6}M")
    
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    num_params = count_parameters(model)
    print("Number of parameters (M):", num_params / 1e6)

    input_tensor = torch.randn(1, 3, 256, 256)  # Example input

    _ = model(input_tensor)  # Warm-up

    start_time = time.time()
    output = model(input_tensor)
    inference_time = time.time() - start_time
    print(f"Inference Time: {inference_time} seconds")
