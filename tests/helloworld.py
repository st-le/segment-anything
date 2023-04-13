import os
import sys
import time
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import cv2
import numpy as np
from sklearn.neighbors import KDTree

import torch
import torch.nn.functional as F

from segment_anything import build_sam, sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser(
    description=("exemplar-based detection")
)
 
parser.add_argument("--use_crop", type=bool, required=False, default=False, help="use crop")
parser.add_argument("--image_path", type=str, required=False, default="/home/quocviet/Downloads/net (9465).jpg",
                     help="Path to either a single input image or folder of images.")

args = parser.parse_args()

def show_mask(mask, ax, rand_color=False):
    if not rand_color:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    else:
        r = np.random.randint(255) / 255
        g = np.random.randint(255) / 255
        b = np.random.randint(255) / 255
        color = np.array([r,g,b,0.6])
    if mask.shape[0] == 3:
        mask = mask[0,:, :]
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

class SAMfeatureExtractor:
    def __init__(self):
        t0 = time.time()
        self.predictor = SamPredictor(build_sam(checkpoint="./sam_vit_h_4b8939.pth"))
        print('load model takes {} seconds'.format(time.time() - t0))

        self.img_embed = None

    def get_exemplar_feat(self, img, points, point_labels):
        predictor = self.predictor
        predictor.set_image(img)
        t0 = time.time()

        """
            1. extract exemplar features by a point-based prompt 
            2. the predicted mask
            3. use a spatial mean-pool to compute the feature vector
        """

        """ masks: full image-size mask
        """
        masks, _, _ = predictor.predict(point_coords=points,
                                        point_labels=point_labels,
                                        return_logits=True)
        print('predict takes {} seconds'.format(time.time() - t0))
        t0 = time.time()

        """ img_embed shape (1, 256, 64, 64)
        """
        img_embed = predictor.get_image_embedding()#.cpu().numpy()
        print('get_image_embedding takes {} seconds'.format(time.time() - t0))

        # visualize the exemplar mask
        plt.figure(figsize=(10,10))
        plt.imshow(img)
        show_mask(masks, plt.gca())
        # plt.axis('on')
        # plt.show()

        masks_resize = F.interpolate(torch.tensor(masks).unsqueeze(0),img_embed.shape[-2:])
        masks_resize_flat = torch.mean(masks_resize,dim=1)
        masks_resize_flat_thr = masks_resize_flat > predictor.model.mask_threshold

        # mask the embedding
        msk_embed = masks_resize_flat_thr.unsqueeze(0) * img_embed
        exemplar_feat = torch.mean(msk_embed, dim=[0,2,3])

        self.img_embed = img_embed
        return exemplar_feat

sfe = SAMfeatureExtractor()

img = cv2.imread(args.image_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
fig, ax = plt.subplots()
ax.imshow(img)
click_x = 0 
click_y = 0
def onclick(event):
    global click_x, click_y
    click_x = event.x
    click_y = event.y
    # print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
    #       ('double' if event.dblclick else 'single', event.button,
    #        event.x, event.y, event.xdata, event.ydata))
cid = fig.canvas.mpl_connect('button_press_event', onclick)
plt.show()

points = np.array([[click_x,click_y]])#np.array([[225,94]])
print('click point:{}'.format(points))

point_labels = np.array([1])
box = np.array([[153,84,225,165]])

""" the exemplar's feature is extracted from the crop box, namely, crop -> feature extract
    instead of feature extract (the whole image) -> crop the feature map
"""
if args.use_crop:
    box0 = box[0,:]
    img = img[box0[1]:box0[3], box0[0]:box0[2], :]

exemplar_feat = sfe.get_exemplar_feat(img, points, point_labels)
predictor = sfe.predictor
img_embed = sfe.img_embed
# detect all instances
# sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
# sam.to(device=device)
t0 = time.time()
# predictor.model.to('cuda')  # OOM
print('to device takes:{}'.format(time.time()-t0))
t0 = time.time()
mask_generator = SamAutomaticMaskGenerator(predictor.model, return_logits=True)
all_inst_masks = mask_generator.generate(img)
print('extract all instance masks take:{}'.format(time.time()-t0))

all_inst_feats = []
for id, m in enumerate(all_inst_masks):
    m = all_inst_masks[0]['mask_logits'][id] #m['segmentation']
    m_resize = F.interpolate(torch.tensor(m).unsqueeze(0).unsqueeze(0),img_embed.shape[-2:])
    m_resize_thr = m_resize > predictor.model.mask_threshold

    # mask the embedding for the instance
    inst_msk_embed = m_resize_thr * img_embed
    inst_feat = torch.mean(inst_msk_embed, dim=[0,2,3])
    all_inst_feats.append(inst_feat.numpy())
all_inst_feats.append(exemplar_feat.numpy())
X = np.stack(all_inst_feats)

t0 = time.time()
kdt = KDTree(X, leaf_size=30, metric='euclidean')
print('build kdtree takes: {}'.format(time.time()-t0))

t0 = time.time()
indices = kdt.query(X, k=30, return_distance=False)
print('query kdtree takes:{}'.format(time.time()-t0))
# print('nn indices:{}'.format(indices))

# collect the nn masks
nn_masks = np.stack([all_inst_masks[id]['segmentation'] for id in indices[-1][1:]])
for i in range(nn_masks.shape[0]):
    show_mask(nn_masks[i], plt.gca())
plt.axis('on')
plt.show()
print('Done.')

