import argparse
import os.path
from pathlib import Path
import torch
import numpy as np
from PIL import Image
from train import get_transform, create_model
import compute_mean_std
from tqdm import tqdm
from torchvision.utils import save_image


def parse_args():
    parser = argparse.ArgumentParser(description='Training models with Pytorch')
    parser.add_argument('--input_dir', default='./DRIVE/gan_out/images', type=str)
    parser.add_argument('--output_dir', default='./DRIVE/gan_out_png/1st_manual', type=str)
    parser.add_argument('--model_path', default='./results/24082022_235033_no_GAN/best_model.pth', type=str)

    return parser.parse_args()


def inference(test_model, im_path_to_infer, test_transform):
    test_model.eval()
    with torch.no_grad():
        img = Image.open(im_path_to_infer)
        trans_im, _ = test_transform(img, Image.fromarray(np.zeros_like(img)))
        trans_im = torch.from_numpy(np.expand_dims(trans_im, 0))
        trans_im.to('cuda')
        output = test_model(trans_im.cuda())
        output[output >= 0.5] = 1
        output[output < 0.5] = 0
    return output.cpu()[0, 0, :, :]


if __name__ == '__main__':
    args = parse_args()
    if not os.path.exists(args.input_dir):
        raise 'invalid input dir'
    if not os.path.exists(args.output_dir):
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    print('computing mean std...')
    #mean, std = compute_mean_std.compute(args.input_dir)
    mean = [0.4863596, 0.26184077, 0.14925525]
    std = [0.34434437, 0.18728918, 0.10615013]
    print(f'mean - {mean}')
    print(f'std - {std}')
    transform = get_transform(train=False, mean=mean, std=std)

    print('loading model...')
    model = create_model(num_classes=1)
    model.to('cuda')
    checkpoint = torch.load(args.model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model'])

    img_names = [i for i in os.listdir(args.input_dir) if i.endswith(".png")]
    img_list = [os.path.join(args.input_dir, i) for i in img_names]

    print(f'{len(img_list)} images found. starting inference')
    for im_name in tqdm(img_names):
        im_path = os.path.join(args.input_dir, im_name)
        res = inference(model, im_path_to_infer=im_path, test_transform=get_transform(train=False, mean=mean, std=std))
        file_name = os.path.splitext(im_path)[0]
        save_image(res, os.path.join(args.output_dir, im_name[:-4] + '.png'))

