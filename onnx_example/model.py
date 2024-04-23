import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.utils.model_zoo as model_zoo
import torchvision.transforms as transforms
from PIL import Image

# Super Resolution model
class SR(nn.Module):
    def __init__(self, upscale_factor, inplace=False):
        super(SR, self).__init__()

        self.relu = nn.ReLU(inplace=inplace)
        self.conv1 = nn.Conv2d(1, 64, (5, 5), (1, 1), (2, 2))
        self.conv2 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        self.conv3 = nn.Conv2d(64, 32, (3, 3), (1, 1), (1, 1))
        self.conv4 = nn.Conv2d(32, upscale_factor ** 2, (3, 3), (1, 1), (1, 1))
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

        self._initialize_weights()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.pixel_shuffle(self.conv4(x))
        return x

    def _initialize_weights(self):
        init.orthogonal_(self.conv1.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv2.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv3.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv4.weight)

if __name__ == '__main__':
    torch_model = SR(upscale_factor=3)

    # weights import
    model_url = 'https://s3.amazonaws.com/pytorch/test_data/export/superres_epoch100-44c6958e.pth'
    torch_model.load_state_dict(model_zoo.load_url(model_url))

    map_location = lambda storage, loc: storage
    if torch.cuda.is_available():
        map_location = None
    torch_model.load_state_dict(model_zoo.load_url(model_url, map_location=map_location)) # Load the model from the URL

    # Input to the model
    torch_model.eval()

    # random input tensor
    x = torch.randn(1, 1, 224, 224, requires_grad=True)
    torch_out = torch_model(x) # model output for checking

    # sample image
    img = Image.open("../assets/cookie.jpg")
    resize = transforms.Resize([224, 224])
    img = resize(img)
    img.save("./results/cookie.jpg")

    img_ycbcr = img.convert('YCbCr')
    img_y, img_cb, img_cr = img_ycbcr.split()

    to_tensor = transforms.ToTensor()
    img_y = to_tensor(img_y)
    img_y.unsqueeze_(0)

    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    img_out_y = to_numpy(torch_model(img_y))
    img_out_y = Image.fromarray(np.uint8((img_out_y[0] * 255.0).clip(0, 255)[0]), mode='L')
    final_img = Image.merge(
        "YCbCr", [
            img_out_y,
            img_cb.resize(img_out_y.size, Image.Resampling.BICUBIC),
            img_cr.resize(img_out_y.size, Image.Resampling.BICUBIC),
        ]).convert("RGB")

    final_img.save("./results/cookie_sr.jpg")

    # ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(img_y)}
    # ort_outs = ort_session.run(None, ort_inputs)
    # img_out_y = ort_outs[0]








