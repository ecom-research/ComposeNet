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

class ImagePreprocessingModule(torch.nn.Module):

    def __init__(self):
        super(ImagePreprocessingModule, self).__init__()
        self.image_features = torch.nn.Sequential(
            torch.nn.BatchNorm1d(512),
            torch.nn.Linear(512, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 512)
        )
        
    def forward(self, x):
        processed = self.image_features(x)
        processed += x
        
        return processed
    
class TextPreprocessingModule(torch.nn.Module):

    def __init__(self):
        super(TextPreprocessingModule, self).__init__()
        self.text_features = torch.nn.Sequential(
            torch.nn.BatchNorm1d(768),
            torch.nn.Linear(768, 768),
            torch.nn.ReLU(),
            torch.nn.Linear(768, 768)
        )
        
    def forward(self, x):
        processed = self.text_features(x)
        processed += x
        
        return processed
    

class ConcatWithLinearModule(torch.nn.Module):

    def __init__(self):
        super(ConcatWithLinearModule, self).__init__()
        self.bert_features = torch.nn.Sequential(
            torch.nn.BatchNorm1d(768),
            torch.nn.Linear(768, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 512)
        )
        self.image_features = torch.nn.Sequential(
            torch.nn.BatchNorm1d(512),
            torch.nn.Linear(512, 512),
            torch.nn.Dropout(p=0.7),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 512)
        )

    def forward(self, x):
        x1 = self.image_features(x[0])
        x2 = self.bert_features(x[1])
        concat_x = torch.cat([x1, x2], 1)

        return concat_x
    

class ImgTextCompositionBase(torch.nn.Module):
    """Base class for image + text composition."""

    def __init__(self):
        super(ImgTextCompositionBase, self).__init__()
        self.normalization_layer = torch_functions.NormalizationLayer(
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
        mod_img1, text_img1, text_img2 = self.compose_img_text_with_extra_data(imgs_query, 
                                                                               modification_texts, 
                                                                               extra_data)
        mod_img1 = self.normalization_layer(mod_img1)
        
        img2 = self.extract_img_feature(imgs_target)
        img2 = self.normalization_layer(img2)
        assert (mod_img1.shape[0] == img2.shape[0] and
                mod_img1.shape[1] == img2.shape[1])
        if soft_triplet_loss:
            return self.compute_soft_triplet_loss_(mod_img1, img2, text_img1, text_img2)
        else:
            return self.compute_batch_based_classification_loss_(mod_img1, img2)

    def compute_soft_triplet_loss_(self, mod_img1, img2, text_img1, text_img2):
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
        
        return self.soft_triplet_loss(torch.cat([mod_img1, img2]), torch.cat([text_img1, text_img2]), triplets)

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
        self.text_model = text_model.TextLSTMModel(
            texts_to_build_vocab=texts,
            word_embed_dim=embed_dim,
            lstm_hidden_dim=embed_dim)
        
        if self.learn_on_regions:
            # overwrite img_model
            print("Using just linear layer for img_model...")
            self.img_model = torch.nn.Sequential(torch.nn.Dropout(p=0.5), 
                                                 EncoderImage(2048, 512, 
                                                              precomp_enc_type='basic', 
                                                              no_imgnorm=True))
#             self.img_model = EncoderImage(2048, 512, precomp_enc_type='weight_norm', no_imgnorm=False).cuda()
            return
        
        # img model
        img_model = torchvision.models.resnet18(pretrained=True)
#         for param in img_model.parameters():
#             param.requires_grad = False
        
        class GlobalAvgPool2d(torch.nn.Module):

            def forward(self, x):
                return F.adaptive_avg_pool2d(x, (1, 1))

        img_model.avgpool = GlobalAvgPool2d() # change shape?
        img_model.fc = torch.nn.Sequential(torch.nn.Linear(512, embed_dim))
#         img_model.fc = torch.nn.Sequential(# torch.nn.Linear(512, 2048),
#                                             # torch.nn.Dropout(p=0.2),
#                                             EncoderImage(2048, 
#                                             512,  
#                                             precomp_enc_type='basic',  
#                                             no_imgnorm=True)) 
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
        self.text_features_mapping = TextPreprocessingModule()
        self.image_features_mapping = ImagePreprocessingModule()
        self.concated_dim = embed_dim + 768
        
        self.image_gated_feature_composer = torch.nn.Sequential(
            ConCatModule(),
            torch.nn.BatchNorm1d(self.concated_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.concated_dim, embed_dim)
        )
        self.image_res_info_composer = torch.nn.Sequential(
            ConCatModule(),
            torch.nn.BatchNorm1d(self.concated_dim), 
            torch.nn.ReLU(),
            torch.nn.Linear(self.concated_dim, self.concated_dim), 
            torch.nn.ReLU(),
            torch.nn.Linear(self.concated_dim, embed_dim)
        )
        
        self.text_gated_feature_composer = torch.nn.Sequential(
            ConCatModule(),
            torch.nn.BatchNorm1d(self.concated_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.concated_dim, 768)
        )
        self.text_res_info_composer = torch.nn.Sequential(
            ConCatModule(),
            torch.nn.BatchNorm1d(self.concated_dim), 
            torch.nn.ReLU(),
            torch.nn.Linear(self.concated_dim, self.concated_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.concated_dim, 768)
        )
        
        
    def compose_img_text_with_extra_data(self, imgs, texts, extra_data):
        batch_size = imgs.shape[0]
        
        img_features_source = self.extract_img_feature(imgs)
        
        source_adjs = [x.split(' ')[0] for x in extra_data[3]]
        target_adjs = texts
        nouns = extra_data[0]
        target_captions = [adj + " " + noun for adj, noun in zip(texts, nouns)]
        source_captions = [adj + " " + noun for adj, noun in zip(source_adjs, nouns)]
        
        joined = source_captions + target_captions
        joined = bc.encode(joined)
        source_captions = torch.from_numpy(joined[:batch_size]).cuda()
        target_captions = torch.from_numpy(joined[batch_size:]).cuda()

        return self.compose_img_text_features(img_features_source, target_captions, 
                                              source_captions)

    def compose_img_text_features(self, img_features, target_captions, source_captions):
        mapped_text_features_target = self.text_features_mapping(target_captions)
        mapped_text_features_source = self.text_features_mapping(source_captions)
        
        mapped_image_features_original = self.image_features_mapping(img_features)
        
        f1_image = self.image_gated_feature_composer((mapped_image_features_original,
                                                      mapped_text_features_target))
        f2_image = self.image_res_info_composer((mapped_image_features_original, 
                                                 mapped_text_features_target))
        
        f_image = torch.sigmoid(f1_image) * img_features * self.a[0] + self.a[1] * f2_image
        
        
        f1_text = self.text_gated_feature_composer((mapped_image_features_original,
                                                    mapped_text_features_target))
        f2_text = self.text_res_info_composer((mapped_image_features_original, 
                                               mapped_text_features_target))
        
        f_text = torch.sigmoid(f1_text) * mapped_text_features_source * self.b[0] + self.b[1] * f2_text
        
        return f_image, f_text, target_captions
