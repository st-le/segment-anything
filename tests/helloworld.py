import os
import sys
import glob
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

click_x = 0
click_y = 0
press_key = 'n'

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
    return ax

class SAM_Feature_Extractor:
    def __init__(self):
        t0 = time.time()
        self.predictor = SamPredictor(build_sam(checkpoint="./sam_vit_h_4b8939.pth"))
        print('load model takes {} seconds'.format(time.time() - t0))

        self.mask_generator = SamAutomaticMaskGenerator(self.predictor.model, return_logits=True)

        self.img_embed = None
        self.exemplar_feat_bank = []
        self.img = None

        self.all_inst_feats = []
        self.all_inst_masks = None

    def set_image(self, img):
        self.img = img

    def detect(self):
        """ just an alias """
        return self.exemplar_based_detect()

    def exemplar_based_detect(self):
        """
            Returns:
                nn_masks : masks of detected instances
        """
        assert self.img is not None, 'need to provide an image first with set_image'        

        self.extract_all_inst_feat()

        # collect the nn masks
        indices = self.train_query_exemplar_knn()
        nn_masks = np.stack([self.all_inst_masks[id]['segmentation'] for id in indices[-1][1:]])
        return nn_masks

    def get_exemplar_feat(self, points, point_labels): # ax=None
        assert self.img is not None
        img = self.img
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
        fig = plt.gcf()
        ax = fig.axes[0] if len(fig.axes) > 0 else fig.add_subplot()
        ax.imshow(img)
        show_mask(masks, ax)
        plt.show()
        # if ax is None:
        #     fig = plt.figure(figsize=(10,10))
        #     aximg = plt.imshow(img)
        #     aximg.ax = show_mask(masks, aximg.ax) # plt.gca()
        # else:
        #     ax.imshow(img)
        #     ax = show_mask(masks, ax)
        # if ax is None:
        #     self.plots = {'fig': fig, 'ax': aximg.ax}
        # else:
        #     self.plots = {'fig': None, 'ax': ax}
        # plt.axis('on')
        # plt.show()

        masks_resize = F.interpolate(torch.tensor(masks).unsqueeze(0),img_embed.shape[-2:])
        masks_resize_flat = torch.mean(masks_resize,dim=1)
        masks_resize_flat_thr = masks_resize_flat > predictor.model.mask_threshold

        # mask the embedding
        msk_embed = masks_resize_flat_thr.unsqueeze(0) * img_embed
        exemplar_feat = torch.mean(msk_embed, dim=[0,2,3])

        self.img_embed = img_embed
        self.exemplar_feat_bank.append(exemplar_feat)
        return exemplar_feat
    
    def extract_all_inst_feat(self):
        img_embed = self.img_embed
        predictor =self.predictor
        t0 = time.time()
        all_inst_masks = self.mask_generator.generate(self.img)
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
        self.all_inst_feats = all_inst_feats
        self.all_inst_masks = all_inst_masks

    def train_query_exemplar_knn(self, k=30):
        for f in self.exemplar_feat_bank:
            self.all_inst_feats.append(f.numpy())
        X = np.stack(self.all_inst_feats)

        t0 = time.time()
        kdt = KDTree(X, leaf_size=30, metric='euclidean')
        print('build kdtree takes: {}'.format(time.time()-t0))

        t0 = time.time()
        while k > X.shape[0]:
            k = int(k / 2)
        indices = kdt.query(X, k=k, return_distance=False)
        print('query kdtree takes:{}'.format(time.time()-t0))
        # print('nn indices:{}'.format(indices))

        return indices

def on_press(event):
    print('press', event.key)
    sys.stdout.flush()
    global press_key
    press_key = event.key

def onclick(event):
    global click_x, click_y
    click_x = event.x
    click_y = event.y
    print('[onclick] click point:{}'.format((click_x, click_y)))
    # print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
    #       ('double' if event.dblclick else 'single', event.button,
    #        event.x, event.y, event.xdata, event.ydata))

def main():
    sfe = SAM_Feature_Extractor()

    img = cv2.imread(args.image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    fig, ax = plt.subplots()
    ax.imshow(img)
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

    sfe.set_image(img)
    exemplar_feat = sfe.get_exemplar_feat(points, point_labels)
    predictor = sfe.predictor
    img_embed = sfe.img_embed
    # detect all instances
    # sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    # sam.to(device=device)
    t0 = time.time()
    # predictor.model.to('cuda')  # OOM
    print('to device takes:{}'.format(time.time()-t0))

    sfe.extract_all_inst_feat()

    # collect the nn masks
    indices = sfe.train_query_exemplar_knn()
    nn_masks = np.stack([sfe.all_inst_masks[id]['segmentation'] for id in indices[-1][1:]])
    for i in range(nn_masks.shape[0]):
        show_mask(nn_masks[i], plt.gca())
    plt.axis('on')
    plt.show()
    print('Done.')

def state_loop(fun):
    def loop(self, *args, **kwargs):
        if self.assets['terminate'] == 0: 
            """ loop program here and wait to do a task """
            self.state_enter()

            # do tasks and state switching
            self.assets = fun(self, *args, **kwargs)

            # possibly add a wait 
            time.sleep(self.assets['sleep_dura'])

            self.state_exit()
            # back to state machine loop
        return self.assets            
    return loop

class State:
    def __init__(self, name, actions=None):
        self.name = name
        self.actions = actions
        
        # assect dict. Asset can be anything e.g. class attributes, functions...
        self.assets = {}
        
        # hold the state machine
        self.sm = self.sm() #None 

    # def loop(self):

    def sm(self):
        return self.assets['state_machine'] if hasattr(self, 'assets') and 'state_machine' in self.assets else None

    def state_enter(self):
        print('Entering state {}...'.format(self.name))

    def state_exit(self):
        print('Exiting state {}...'.format(self.name))

class InspectState(State):
    def __init__(self,name):
        super().__init__(name)

    @state_loop
    def loop(self, **kwargs):

        # show the result & get the feedback key
        # if self.assets['plots'] is None or 'plots' not in self.assets or self.assets['plots']['fig'] is None or self.assets['plots']['ax'] is None:
        #     if self.assets['plots'] is None:
        #         fig, ax = plt.subplots()
        #     elif self.assets['plots']['ax'] is None:
        #         raise ValueError('ax cannot be None')
        #     else:
        #         fig = self.assets['plots']['fig'] if self.assets['plots']['fig'] is not None else plt.gcf()
        #         ax = self.assets['plots']['ax']
        # else:
        #     fig, ax = list(self.assets['plots'].values())
        fig = plt.gcf()
        ax = fig.axes[0] if len(fig.axes) > 0 else fig.add_subplot()

        fig.canvas.mpl_connect('key_press_event', on_press)

        ax.imshow(self.assets['image'])
        ax.text(20, 20, "The segment is good enough (y/n)?")

        nn_masks = self.assets['segment_results']
        for i in range(nn_masks.shape[0]):
            ax = show_mask(nn_masks[i], ax) # plt.gca()
        # plt.axis('on')
        plt.show()
        # plt.waitforbuttonpress()

        global press_key
        if not hasattr(self,'sm') or self.sm is None:
            self.sm = self.assets['state_machine']

        if press_key.lower() == 'y':
            # switch state
            self.sm.cur_state = self.assets['states']['WAIT']

        elif press_key.lower() == 'n':
            self.assets = extract_exemplar_feat_userinput(self)
            self.sm.cur_state = self.assets['states']['WAIT']  # switch state

        else:
            print("Key not support. Consider the answer was 'y'")
            # switch state
            self.sm.cur_state = self.assets['states']['WAIT']

        # self.assets['plots'] = {'fig': fig, 'ax': ax}
        self.assets['state_machine'] = self.sm
        return self.assets


def extract_exemplar_feat_userinput(state):
    print('Please click the point to extract the exemplar feature')
    # if state.assets['plots'] is None or 'plots' not in state.assets or state.assets['plots']['fig'] is None or state.assets['plots']['ax'] is None:
    #     if state.assets['plots'] is None:
    #         fig, ax = plt.subplots()
    #     elif state.assets['plots']['ax'] is None:
    #         raise ValueError('ax cannot be None')
    #     else:
    #         fig = state.assets['plots']['fig'] if state.assets['plots']['fig'] is not None else plt.gcf()
    #         ax = state.assets['plots']['ax']
    # else:
    #     fig, ax = list(state.assets['plots'].values())
    fig = plt.gcf()
    # initialize an axes of the fig
    ax = fig.axes[0] if len(fig.axes) > 0 else fig.add_subplot()
    if not hasattr(state,'img'):
        if "image" in state.assets:
            state.img = state.assets['image']
        else:
            raise ValueError("No image in the state assets")
    ax.imshow(state.img)
    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()
    # plt.waitforbuttonpress()

    points = np.array([[click_x,click_y]])
    print('click point:{}'.format(points))

    point_labels = np.array([1])
    box = np.array([[153,84,225,165]])
    state.assets['predictor'].img = state.img
    
    state.assets['extract_exemplar_feat'](points, point_labels) # a shared function
    # state.assets['plots'] = state.assets['predictor'].plots
    return state.assets

class WaitState(State):
    def __init__(self, name):
        super().__init__(name)
        self.action_names = ['detect', 'set_image']

    @state_loop
    def loop(self, **kwargs):

        img_buffer = self.assets['img_buffer']
        img_idx = self.assets['img_idx']
        self.sm = self.assets['state_machine']

        img_idx += 1
        img = cv2.imread(img_buffer[img_idx])
        img_bgr = img.copy()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.assets['image'] = img

        self.img = self.assets['image']
        print('\ninput keys: q to quit, d to detect, s to set image')
        cv2.imshow('Continuous Learner App', img_bgr)
        c = cv2.waitKey(0)
        c = chr(c)
        if c.lower()=='q': 
            print('Program exit...')
            self.assets['terminate'] = 1
        elif c.lower()=='d':
            self.detect()
        elif c.lower()=='s':
            self.set_image(self.img)

        self.assets['img_idx'] = img_idx
        self.assets['state_machine'] = self.sm
        return self.assets
        
    def detect(self):
        if len(self.assets['predictor'].exemplar_feat_bank) == 0:
            # fig, ax = plt.subplots()
            # ax.imshow(self.img)
            # cid = fig.canvas.mpl_connect('button_press_event', onclick)
            # plt.show()

            # points = np.array([[click_x,click_y]])
            # print('click point:{}'.format(points))

            # point_labels = np.array([1])
            # box = np.array([[153,84,225,165]])
            # if self.assets['predictor'].img is None:
            #     self.assets['predictor'].img = self.img
            # self.assets['extract_exemplar_feat'](points, point_labels) # a shared function
            self.assets = extract_exemplar_feat_userinput(self)

            # switch state
            self.sm.cur_state = self.assets['states']['WAIT']
        else:
            self.assets['predictor'].set_image(self.img)

            nn_masks = self.assets['predictor'].detect()
            self.assets['segment_results'] = nn_masks

            # switch state
            self.sm.cur_state = self.assets['states']['INSPECT']

    def set_image(self, img):
        self.assets['predictor'].set_image(img)
        
        # switch state
        self.sm.cur_state = self.assets['states']['WAIT']

class StateMachine:
    def __init__(self):
        self.state_names = [] 
        self.states = {}
        self.cur_state = None
        self.assets = {}

    def loop(self):
        while not self.assets['terminate']:
            self.cur_state.assets = self.assets
            self.assets = self.cur_state.loop()

StateFactory = {
    'WAIT': WaitState,
    'INSPECT': InspectState
}

class Exemplar_Detector_App(StateMachine):
    def __init__(self):
        """ 
            a state machine app to define a workflow for the exemplar learner
            for workflow, see link: shorturl.at/qEFO2
        """
        super().__init__()
        self.state_names = ['WAIT', 'INSPECT'] 
        self.states = {sn:StateFactory[sn](sn) for sn in self.state_names}

        # prepare assets
        sfe = SAM_Feature_Extractor()
        self.assets['predictor'] = sfe
        self.assets['states'] = self.states
        self.assets['sleep_dura'] = 0.010  #sec
        self.assets['state_machine'] = self
        self.assets['extract_exemplar_feat'] = sfe.get_exemplar_feat
        # self.assets['plots'] = None

        self.image_dir = '/home/quocviet/Downloads/OSCD_val2017'

        img_buffer = [g for g in glob.glob(os.path.join(self.image_dir,'*.jpg'))]
        assert len(img_buffer) > 0
        self.assets['img_buffer'] = img_buffer
        self.assets['img_idx'] = -1

        # terminate variable for the whole state machine
        self.assets['terminate'] = 0

        # switch state
        self.cur_state = self.states['WAIT']       

        self.loop()

def main_app():
    app = Exemplar_Detector_App()
    app.loop()

if __name__ == '__main__':
    # main()
    main_app()