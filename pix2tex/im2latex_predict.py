import yaml
import base64
import cv2
import numpy as np
import torch
import os

from transformers import PreTrainedTokenizerFast
from pix2tex.dataset.transforms import test_transform
from pix2tex.models import get_model
from pix2tex.utils import post_process, token2str
from munch import Munch

#get path for model, config and tokenizer
PATH_MODEL = './models/ViT-SingleGPU_e11_step9979_test.pth'
PATH_CONFIG = "./models/config.yaml"
PATH_TOKENIZER = './models/weai_tokenizer.json'

with open(PATH_CONFIG, "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
args = Munch(config)

args.device = 'cpu'
args.wandb = False

tokenizer = PreTrainedTokenizerFast(tokenizer_file=PATH_TOKENIZER)

model = get_model(args)
model.load_state_dict(torch.load(PATH_MODEL, map_location=args.device))
model.eval()

def pad_w(img, pad_w):
    left_pad =32
    right_pad = pad_w + 16
    return np.copy(np.pad(img, ((0, 0), (left_pad, right_pad), (0,0)), mode='constant', constant_values=255)) # type: ignore

def pad_h(img, pad_h):
    top_pad = 32
    bottom_pad = pad_h + 16
    return np.copy(np.pad(img, ((top_pad, bottom_pad), (0, 0), (0,0)), mode='constant', constant_values=255)) # type: ignore

def predict(img):
    img = base64.b64decode(img)
    im = cv2.imdecode(np.frombuffer(img, np.uint8), cv2.IMREAD_UNCHANGED)
    
    height, width = im.shape[0], im.shape[1]

    pad_width, pad_height = 0, 0
    if height % 16 != 0:
        pad_height = 16 - (height % 16)
        im = pad_h(im, pad_height)

    if width % 16 != 0:
        pad_width = 16 - (width % 16)
        im = pad_w(im, pad_width)

    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    img_pred = test_transform(image=im)['image'][:1].unsqueeze(0)

    dec = model.generate(img_pred.to(args.device), temperature=args.get('temperature', .2)) # return decoded sequence of tokens
    # print(dec)
    pred = post_process(token2str(dec, tokenizer)[0]) #str
    # print(" --- returned equation: ", pred)
    return pred