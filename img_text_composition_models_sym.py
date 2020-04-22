# Copyright 2019 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Models for Text and Image Composition."""
from collections import OrderedDict

import numpy as np
import torch
import torchvision
import torch.nn.functional as F
import text_model
import re
import torch_functions
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn.utils.weight_norm import weight_norm
import torch.backends.cudnn as cudnn
from torch.nn.utils.clip_grad import clip_grad_norm
from bert_serving.client import BertClient
from gensim.models import KeyedVectors

# wv = KeyedVectors.load('../tirg-with-scan/wordvectors-300.kv', mmap='r')

bc = BertClient()

class ConCatModule(torch.nn.Module):

    def __init__(self):
        super(ConCatModule, self).__init__()

    def forward(self, x):
        x = torch.cat(x, 1)

        return x


    
class ConcatWithLinearAndResidualModule(torch.nn.Module):

    def __init__(self):
        super(ConcatWithLinearAndResidualModule, self).__init__()
        self.image_features_mapping = torch.nn.Sequential(
            torch.nn.BatchNorm1d(512),
            torch.nn.Linear(512, 512),
            torch.nn.Dropout(p=0.5),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 512)
        )
        self.text_features_mapping = torch.nn.Sequential(
            torch.nn.BatchNorm1d(768),
            torch.nn.Linear(768, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 512)
        )
        

    def forward(self, x):
        image_res = x[0]
        text_res = x[1]
        
        # text_res = self.text_residual_mapping(x[1])
        
        x0 = self.image_features_mapping(x[0])
        x1 = self.text_features_mapping(x[1])
        
        x0 += image_res
        # x1 += text_res
        
        concat_x = torch.cat([x0, x1], 1)

        return concat_x
    
class w2vConcatWithLinearModule(torch.nn.Module):

    def __init__(self):
        super(w2vConcatWithLinearModule, self).__init__()
        self.image_features = torch.nn.Sequential(
            torch.nn.BatchNorm1d(512),
            torch.nn.Linear(512, 512),
            torch.nn.Dropout(p=0.5),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 512)
        )

    def forward(self, x):
        x1 = self.image_features(x[0])
        x2 = x[1]
        
        concat_x = torch.cat([x1, x2], 1)

        return concat_x
    

class ImgTextCompositionBase(torch.nn.Module):
    """Base class for image + text composition."""

    def __init__(self):
        super(ImgTextCompositionBase, self).__init__()
        self.normalization_layer = torch_functions.NormalizationLayer(
            normalize_scale=4.0, learn_scale=True)
        self.text_normalization_layer = torch_functions.NormalizationLayer(
            normalize_scale=4.0, learn_scale=True)
        self.soft_triplet_loss = torch_functions.TripletLoss()

    def extract_img_feature(self, imgs):
        raise NotImplementedError

    def extract_text_feature(self, texts):
        raise NotImplementedError

    def compose_img_text(self, imgs, texts):
        raise NotImplementedError

    def compute_loss(self,
                     imgs_query,
                     modification_texts,
                     imgs_target,
                     soft_triplet_loss=True):
        mod_img1 = self.compose_img_text(imgs_query, modification_texts)  # ids
        mod_img1 = self.normalization_layer(mod_img1)
        
        img2 = self.extract_img_feature(imgs_target)
        img2 = self.normalization_layer(img2)
        assert (mod_img1.shape[0] == img2.shape[0] and
                mod_img1.shape[1] == img2.shape[1])
        if soft_triplet_loss:
            return self.compute_soft_triplet_loss_(mod_img1, img2)
        else:
            return self.compute_batch_based_classification_loss_(mod_img1, img2)
        
    def compute_loss_with_extra_data(self,
                                  imgs_query,
                                  modification_texts,
                                  imgs_target,
                                  extra_data,
                                  soft_triplet_loss=True):
        mod_img1, img2, mod_img2, img1, f_image1_decoded, img2_copy = self.compose_img_text_with_extra_data(imgs_query, 
                                                                              modification_texts, 
                                                                              extra_data, 
                                                                              imgs_target)
        img1 = self.normalization_layer(img1)
        img2 = self.normalization_layer(img2)
        
        mod_img1 = self.normalization_layer(mod_img1)
        mod_img2 = self.normalization_layer(mod_img2)
        
        f_image1_decoded = self.normalization_layer(f_image1_decoded)
        
        assert (mod_img1.shape[0] == img2.shape[0] and
                mod_img1.shape[1] == img2.shape[1])
        if soft_triplet_loss:
            return self.compute_soft_triplet_loss_(mod_img1, img2), self.compute_soft_triplet_loss_(mod_img2, img1)
        # , F.mse_loss(f_image1_decoded, img2).cpu()
        else:
            return self.compute_batch_based_classification_loss_(mod_img1, img2)

    def compute_soft_triplet_loss_(self, mod_img1, img2):
        triplets = []
        labels = range(mod_img1.shape[0]) + range(img2.shape[0])
        for i in range(len(labels)):
            triplets_i = []
            for j in range(len(labels)):
                if labels[i] == labels[j] and i != j:
                    for k in range(len(labels)):
                        if labels[i] != labels[k]:
                            triplets_i.append([i, j, k])
            np.random.shuffle(triplets_i)
            triplets += triplets_i[:3]
        assert (triplets and len(triplets) < 2000)
        
        return self.soft_triplet_loss(torch.cat([mod_img1, img2]), triplets)

    def compute_batch_based_classification_loss_(self, mod_img1, img2):
        x = torch.mm(mod_img1, img2.transpose(0, 1))
        labels = torch.tensor(range(x.shape[0])).long()
        labels = torch.autograd.Variable(labels).cuda()
        return F.cross_entropy(x, labels)


class ImgEncoderTextEncoderBase(ImgTextCompositionBase):
    """Base class for image and text encoder."""

    def __init__(self, texts, embed_dim, learn_on_regions):
        super(ImgEncoderTextEncoderBase, self).__init__()
        self.learn_on_regions = learn_on_regions

        # text model
#         self.text_model = text_model.TextLSTMModel(
#             texts_to_build_vocab=texts,
#             word_embed_dim=embed_dim,
#             lstm_hidden_dim=embed_dim)
        
        if self.learn_on_regions:
            # overwrite img_model
            print("Using just linear layer for img_model...")
            self.img_model = torch.nn.Sequential(torch.nn.Dropout(p=0.5), 
                                                 EncoderImage(2048, 512, 
                                                              precomp_enc_type='basic', 
                                                              no_imgnorm=True))
            return
        
        # img model
        img_model = torchvision.models.resnet18(pretrained=True)
        
        class GlobalAvgPool2d(torch.nn.Module):

            def forward(self, x):
                return F.adaptive_avg_pool2d(x, (1, 1))

        img_model.avgpool = GlobalAvgPool2d() # change shape?
        img_model.fc = torch.nn.Sequential(torch.nn.Linear(512, embed_dim))
        self.img_model = img_model

    def extract_img_feature(self, imgs):
        return self.img_model(imgs)
            

    def extract_text_feature(self, texts):
        return self.text_model(texts)


class SimpleModelImageOnly(ImgEncoderTextEncoderBase):

    def compose_img_text(self, imgs, texts):
        return self.extract_img_feature(imgs)


class SimpleModelTextOnly(ImgEncoderTextEncoderBase):

    def compose_img_text(self, imgs, texts):
        return self.extract_text_feature(texts)


class Concat(ImgEncoderTextEncoderBase):
    """Concatenation model."""

    def __init__(self, texts, embed_dim, learn_on_regions):
        super(Concat, self).__init__(texts, embed_dim)

        # composer
        class Composer(torch.nn.Module):
            """Inner composer class."""

            def __init__(self):
                super(Composer, self).__init__()
                self.m = torch.nn.Sequential(
                    torch.nn.BatchNorm1d(2 * embed_dim), torch.nn.ReLU(),
                    torch.nn.Linear(2 * embed_dim, 2 * embed_dim),
                    torch.nn.BatchNorm1d(2 * embed_dim), torch.nn.ReLU(),
                    torch.nn.Dropout(0.1), torch.nn.Linear(2 * embed_dim, embed_dim))

            def forward(self, x):
                f = torch.cat(x, dim=1)
                f = self.m(f)
                return f

        self.composer = Composer()

    def compose_img_text(self, imgs, texts):
        img_features = self.extract_img_feature(imgs)
        text_features = self.extract_text_feature(texts)
        return self.compose_img_text_features(img_features, text_features)

    def compose_img_text_features(self, img_features, text_features):
        return self.composer((img_features, text_features))


class TIRG(ImgEncoderTextEncoderBase):
    """The TIGR model.

    The method is described in
    Nam Vo, Lu Jiang, Chen Sun, Kevin Murphy, Li-Jia Li, Li Fei-Fei, James Hays.
    "Composing Text and Image for Image Retrieval - An Empirical Odyssey"
    CVPR 2019. arXiv:1812.07119
    """

    def __init__(self, texts, embed_dim, learn_on_regions):
        super(TIRG, self).__init__(texts, embed_dim, learn_on_regions)

        self.a = torch.nn.Parameter(torch.tensor([1.0, 10.0, 1.0, 1.0]))
        self.gated_feature_composer = torch.nn.Sequential(
            ConCatModule(), 
            torch.nn.BatchNorm1d(2 * embed_dim),
            # torch.nn.Dropout(p=0.2),
            torch.nn.ReLU(),
            torch.nn.Linear(2 * embed_dim, embed_dim))
        self.res_info_composer = torch.nn.Sequential(
            ConCatModule(), 
            torch.nn.BatchNorm1d(2 * embed_dim), 
            # torch.nn.Dropout(p=0.2),
            torch.nn.ReLU(),
            torch.nn.Linear(2 * embed_dim, 2 * embed_dim), 
            torch.nn.ReLU(),
            torch.nn.Linear(2 * embed_dim, embed_dim))

    def compose_img_text(self, imgs, texts):
        img_features = self.extract_img_feature(imgs)

        text_features = self.extract_text_feature(texts)
        return self.compose_img_text_features(img_features, text_features)

    def compose_img_text_features(self, img_features, text_features):
        f1 = self.gated_feature_composer((img_features, text_features))
        f2 = self.res_info_composer((img_features, text_features))
        f = torch.sigmoid(f1) * img_features * self.a[0] + f2 * self.a[1]
        return f

import torch
from torch.autograd import Function

class L1Penalty(Function):

    @staticmethod
    def forward(ctx, input, l1weight):
        ctx.save_for_backward(input)
        ctx.l1weight = l1weight
        return input

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_variables
        grad_input = input.clone().sign().mul(ctx.l1weight)
        grad_input += grad_output
        return grad_input, None
    
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.main = [
            nn.Linear(64, 128),
            nn.ReLU(True),
            nn.Linear(128, 256),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.ReLU(True),
            nn.Linear(512, 512)
        ]
        for idx, module in enumerate(self.main):
            self.add_module(str(idx), module)

    def forward(self, x):
        for layer in self.main:
            x = layer(x)
        return x
# [(torch.Size([512, 512]), 'encoder.0.weight'),
#  (torch.Size([512]), 'encoder.0.bias'),
#  (torch.Size([256, 512]), 'encoder.2.weight'),
#  (torch.Size([256]), 'encoder.2.bias'),
#  (torch.Size([128, 256]), 'encoder.4.weight'),
#  (torch.Size([128]), 'encoder.4.bias'),
#  (torch.Size([64, 128]), 'encoder.6.weight'),
#  (torch.Size([64]), 'encoder.6.bias')]
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.main = [
            # input is (nc) x 64 x 64
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, 256),
            nn.ReLU(True), 
            nn.Linear(256, 128),
            nn.ReLU(True), 
            nn.Linear(128, 64)
        ]

        for idx, module in enumerate(self.main):
            self.add_module(str(idx), module)

    def forward(self, x):
        for layer in self.main:
            x = layer(x)
        x = L1Penalty.apply(x, 0.01)
        return x
    

class TIRGEvolved(ImgEncoderTextEncoderBase):
    """The TIGR model.

    The method is described in
    Nam Vo, Lu Jiang, Chen Sun, Kevin Murphy, Li-Jia Li, Li Fei-Fei, James Hays.
    "Composing Text and Image for Image Retrieval - An Empirical Odyssey"
    CVPR 2019. arXiv:1812.07119
    """

    def __init__(self, texts, embed_dim, learn_on_regions):
        super(TIRGEvolved, self).__init__(texts, embed_dim, learn_on_regions)
        
        self.a = torch.nn.Parameter(torch.tensor([1.0, 10.0, 1.0, 1.0]))
        self.b = torch.nn.Parameter(torch.tensor([1.0, 10.0, 1.0, 1.0])) # change to 3.0 10.0 ?
        self.c = torch.nn.Parameter(torch.tensor([1.0, 10.0, 1.0, 1.0]))
#        self.wv = KeyedVectors.load('../tirg-with-scan/wordvectors-300.kv', mmap='r')
        self.encoder = Encoder().cuda()
        self.decoder = Decoder().cuda()
        # self.encoder.load_state_dict(torch.load('Q_latest.pth'))
        # self.decoder.load_state_dict(torch.load('P_latest.pth'))
        
        self.encoder.eval()
        self.decoder.eval()
    
        self.text_features_mapping = torch.nn.Sequential(
            torch.nn.BatchNorm1d(768),
            torch.nn.Linear(768, 768),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(768),
            torch.nn.Linear(768, 512)
        )
        
        self.image_features_mapping = torch.nn.Sequential(
            torch.nn.BatchNorm1d(512),
            torch.nn.Linear(512, 512),
            torch.nn.Dropout(p=0.7),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(512),
            torch.nn.Linear(512, 512)
        ) 
        
        self.image_gated_feature_composer = torch.nn.Sequential(
            ConCatModule(),
            torch.nn.BatchNorm1d(2 * embed_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(2 * embed_dim, embed_dim)
        )
        self.image_res_info_composer = torch.nn.Sequential(
            ConCatModule(),
            torch.nn.BatchNorm1d(2 * embed_dim), 
            torch.nn.ReLU(),
            torch.nn.Linear(2 * embed_dim, 2 * embed_dim), 
            torch.nn.ReLU(),
            torch.nn.Linear(2 * embed_dim, embed_dim)
        )

        self.text_gated_feature_composer = torch.nn.Sequential(
            ConCatModule(),
            torch.nn.BatchNorm1d(2 * embed_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(2 * embed_dim, 768)
        )
        self.text_res_info_composer = torch.nn.Sequential(
            ConCatModule(),
            torch.nn.BatchNorm1d(2 * embed_dim), 
            torch.nn.ReLU(),
            torch.nn.Linear(2 * embed_dim, 2 * embed_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(2 * embed_dim, 768)
        )

        
    def extract_coord_info(self, imgs):
        return self.coord_extractor(imgs)
   
    def extract_attention_info(self, texts, scene_embs):
        attention = Attention(768)
        output, weights = attention(texts, # query
                                    scene_embs # context
                                    )
        
        return output
    
    def get_wv_reprs(self, s):
        v = np.zeros(300)
        for w in re.split('\W+', s):
            if w == 'to':
                continue
            if w == 'botm':
                w = 'bottom'
            try:
                v += wv.get_vector(w)
            except KeyError as e:
                print(e, s)
        return v
    
    def get_w2v_for_lst(self, l):
        embs = []
        for i in l:
            embs.append(self.get_wv_reprs(i))
        return embs
    
    def apply_another_attention(self, scenes, queries, raw_feature_norm):
        queries = np.array(self.get_w2v_for_lst(queries))
        scenes = self.pad_descriptions(scenes)
        scenes_summed = np.sum(scenes, axis=1)

        queries_summed_scenes = queries + scenes_summed
        
        queries = torch.from_numpy(queries).float().unsqueeze(1).to('cuda') 
        scenes_wv = torch.from_numpy(scenes).float().to('cuda')
        queries_summed_scenes = torch.from_numpy(queries_summed_scenes).float().to('cuda')
        
        weighted, attT = func_attention(queries, scenes_wv, raw_feature_norm)
        
        return weighted.squeeze(1), queries_summed_scenes
    
    def pad_descriptions(self, extra_data):
        from itertools import chain
        
        # flatten extra_data
        flattened = list(chain.from_iterable(extra_data))
        try:
            encodings = np.array(self.get_w2v_for_lst(flattened))
            # encodings = bc.encode(flattened)
            # encodings = self.extract_text_feature(flattened)
        except ValueError as e:
            print(e)
            print(flattened)
            print('got extra data as', extra_data)
        lengths = [len(x) for x in extra_data]
        max_len = max(lengths)
        
        # get intervals
        intervals = []
        start = 0
        
        for l in lengths:
            interval = (start, start + l)
            intervals.append(interval)
            start += l
        
        padded_encods = []
        for i in intervals:
            e = encodings[i[0]:i[1]]
            current_len = e.shape[0]
            padded_e = np.pad(e, ((0, max_len - current_len), (0, 0)), 'constant', constant_values=(0, )) # np
            # padded_e = torch.nn.functional.pad(e, (0, 0, 0, max_len - current_len), 'constant', value=0)
            padded_encods.append(padded_e)
        
        padded_encods = np.array(padded_encods) # hard coded

        return padded_encods
            
    def get_w2v_representation(self, source_adjs, nouns, target_adjs, batch_size):
        arithmetics = np.zeros((batch_size, 300))
        target = np.zeros((batch_size, 300))
        for i in range(batch_size):
            arithmetics[i] = self.wv[nouns[i]] + self.wv[target_adjs[i]] - self.wv[source_adjs[i]]
            target[i] = self.wv[nouns[i]] + self.wv[target_adjs[i]]

        return torch.from_numpy(arithmetics).float().cuda(), torch.from_numpy(target).float().cuda()

    def compose_img_text_with_extra_data(self, imgs, texts, extra_data, imgs_target):
        batch_size = imgs.shape[0]
        
        img1_features = self.extract_img_feature(imgs)
        if not isinstance(imgs_target, list): # if it is train
            img2_features = self.extract_img_feature(imgs_target)
        else:
            img2_features = []
        # switch_dct = {'remove': 'add', 'add': 'remove'}

        if len(extra_data) > 2: # mitstates
            source_adjs = [x.split(' ')[0] for x in extra_data[3]]
            target_adjs = texts
            nouns = extra_data[0]
            
            target_captions = [adj + " " + noun for adj, noun in zip(texts, nouns)]
            source_captions = [adj + " " + noun for adj, noun in zip(source_adjs, nouns)]
            # print(target_captions[0],source_captions[0])
            # self.target_captions = self.extract_text_feature(text_features)
        elif len(extra_data) == 2: # css
            target_captions = [" ".join(x) if len(x) > 0 else "No objects" for x in extra_data[0]]
            source_captions = [" ".join(x) if len(x) > 0 else "No objects" for x in extra_data[1]]
                
            text_features = texts
            self.target_captions = self.extract_text_feature(texts)
            # fix unknown words to w2v, get dict with remove-add mapping
        else:
            text_features = extra_data[0] # or texts, try
        joined = target_captions + source_captions
        joined = bc.encode(joined)
        
        target_captions = torch.from_numpy(joined[:batch_size]).cuda()
        source_captions = torch.from_numpy(joined[batch_size:]).cuda()

        return self.compose_img_text_features(img1_features, source_captions, 
                                              target_captions, img2_features)

    def compose_img_text_features(self, img1_features, source_captions, target_captions, img2_features):
        # untouched img1 and img2, normalized
        mapped_image_features_source = self.image_features_mapping(img1_features)
        mapped_text_features_target = self.text_features_mapping(target_captions)
        
        if not isinstance(img2_features, list): # if it is train
            mapped_text_features_source = self.text_features_mapping(source_captions)
            mapped_image_features_target = self.image_features_mapping(img2_features)
        else:
            mapped_image_features_target = []
        
        # source_image + target_text => target_image
        f1_image1 = self.image_gated_feature_composer((mapped_image_features_source,
                                                       mapped_text_features_target))
        f2_image1 = self.image_res_info_composer((mapped_image_features_source, 
                                                  mapped_text_features_target))
        
        f_image1 = torch.sigmoid(f1_image1) * img1_features * self.a[0] + self.a[1] * f2_image1

        
        # target_image + source_text => source_image, train mode only
        if not isinstance(mapped_image_features_target, list):
            f_image2 = None
            f1_image2 = self.image_gated_feature_composer((mapped_image_features_target,
                                                           mapped_text_features_source))
            f2_image2 = self.image_res_info_composer((mapped_image_features_target, 
                                                      mapped_text_features_source))
            f_image2 = torch.sigmoid(f1_image2) * img2_features * self.c[0] + self.c[1] * f2_image2

        else:
            f_image2 = None
            f_image1_decoded = None
            
        # autoencoder part
        z_sample = self.encoder(f_image1) # encoder
        f_image1_decoded = self.decoder(z_sample) # decoder
            
        
        return f_image1, img2_features, f_image2, img1_features, f_image1_decoded, f_image1 
