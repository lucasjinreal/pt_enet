"""
Demo file shows how to inference semantic segmentation
with cityscapes or other trained model in ENet method

"""
import torch
import torchvision.transforms as transforms
from torch.autograd import Variable

from models.enet import ENet
from alfred.dl.torch.common import device
import cv2
from PIL import Image
import numpy as np
from alfred.vis.image.get_dataset_colormap import label_to_color_image
from alfred.dl.inference.image_inference import ImageInferEngine


class ENetDemo(ImageInferEngine):

    def __init__(self, f, model_path):
        super(ENetDemo, self).__init__(f=f)

        self.target_size = (512, 1024)
        self.model_path = model_path
        self.num_classes = 20

        self.image_transform = transforms.Compose(
            [transforms.Resize(self.target_size),
             transforms.ToTensor()])

        self._init_model()

    def _init_model(self):
        self.model = ENet(self.num_classes).to(device)
        checkpoint = torch.load(self.model_path)
        self.model.load_state_dict(checkpoint['state_dict'])
        print('Model loaded!')

    def solve_a_image(self, img):
        images = Variable(self.image_transform(Image.fromarray(img)).to(device).unsqueeze(0))
        predictions = self.model(images)
        _, predictions = torch.max(predictions.data, 1)
        prediction = predictions.cpu().numpy()[0] - 1
        return prediction

    def vis_result(self, img, net_out):
        mask_color = np.asarray(label_to_color_image(net_out, 'cityscapes'), dtype=np.uint8)
        frame = cv2.resize(img, (self.target_size[1], self.target_size[0]))
        # mask_color = cv2.resize(mask_color, (frame.shape[1], frame.shape[0]))
        res = cv2.addWeighted(frame, 0.5, mask_color, 0.7, 1)
        return res


if __name__ == '__main__':
    v_f = '/media/jintain/sg/permanent/datasets/Cityscapes/videos/combined_stuttgart_01.mp4'
    enet_seg = ENetDemo(f=v_f, model_path='save/ENet_cityscapes_mine.pth')
    enet_seg.run()

