import argparse
import os, sys
import torch, json
import numpy as np

from main import build_model_main
from util.slconfig import SLConfig
from datasets import build_dataset
from util.visualizer import COCOVisualizer
from util import box_ops
from PIL import Image
from pathlib import Path
import datasets.transforms as T


class Colors:
    # Ultralytics color palette https://ultralytics.com/
    def __init__(self):
        # hex = matplotlib.colors.TABLEAU_COLORS.values()
        hexs = ('FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231', '48F90A', '92CC17', '3DDB86', '1A9334', '00D4BB',
                '2C99A8', '00C2FF', '344593', '6473FF', '0018EC', '8438FF', '520085', 'CB38FF', 'FF95C8', 'FF37C7')
        self.palette = [self.hex2rgb(f'#{c}') for c in hexs]
        self.n = len(self.palette)

    def __call__(self, i, bgr=False):
        c = self.palette[int(i) % self.n]
        return (c[2]/255.0, c[1]/255.0, c[0]/255.0) if bgr else c

    @staticmethod
    def hex2rgb(h):  # rgb order (PIL)
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))


colors = Colors()


def detect(img_filename, dir=None):
    if dir is not None:
        image = Image.open(os.path.join(dir, img_filename)).convert("RGB")  # load image
    else:
        image = Image.open(img_filename).convert("RGB")
    # transform images
    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image, _ = transform(image, None)
    if device == torch.device('cuda'):
        output = model.cuda()(image[None].cuda())
        output = postprocessors['bbox'](output, torch.Tensor([[1.0, 1.0]]).cuda())[0]
    else:
        output = model(image[None])
        output = postprocessors['bbox'](output, torch.Tensor([[1.0, 1.0]]))[0]

    thershold = 0.3
    vslzr = COCOVisualizer()

    scores = output['scores']
    labels = output['labels']
    boxes = box_ops.box_xyxy_to_cxcywh(output['boxes'])
    select_mask = scores > thershold

    box_label = [id2name[int(item)] for item in labels[select_mask]]
    box_colors = [colors(int(item), bgr=True) for item in labels[select_mask]]
    pred_dict = {
        'image_id': 0,
        'boxes': boxes[select_mask],
        'size': torch.Tensor([image.shape[1], image.shape[2]]),
        'box_label': box_label,
        'box_colors': box_colors
    }
    vslzr.visualize(image, pred_dict, caption=img_filename, savedir=args_parser.output_dir, dpi=100)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DETR training and evaluation script')
    parser.add_argument("--model_config_path", type=str)
    parser.add_argument("--model_checkpoint_path", type=str)
    parser.add_argument("--image_input", type=str)
    parser.add_argument("--output_dir", type=str)
    args_parser = parser.parse_args()
    # args.device = device
    if args_parser.output_dir:
        Path(args_parser.output_dir).mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    model_config_path = args_parser.model_config_path
    model_checkpoint_path = args_parser.model_checkpoint_path
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    model, criterion, postprocessors = build_model_main(args)
    checkpoint = torch.load(model_checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    _ = model.eval()

    with open('util/visdrone_id2name.json') as f:
        id2name = json.load(f)
        id2name = {int(k): v for k, v in id2name.items()}
        # name2id = {v: int(k) for k, v in id2name.items()}

    if os.path.isdir(args_parser.image_input):
        for img_filename in os.listdir(args_parser.image_input):
            detect(img_filename, args_parser.image_input)
    else:
        detect(args_parser.image_input)
