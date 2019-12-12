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
import torch_functions
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn.utils.weight_norm import weight_norm
import torch.backends.cudnn as cudnn
from torch.nn.utils.clip_grad import clip_grad_norm


class ConCatModule(torch.nn.Module):

    def __init__(self):
        super(ConCatModule, self).__init__()

    def forward(self, x):
        x = torch.cat(x, 1)
        # print(x.shape)
        return x


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

    def compute_loss_with_regions(self,
                                  imgs_query,
                                  modification_texts,
                                  imgs_target,
                                  scan_info,
                                  soft_triplet_loss=True):
        mod_img1 = self.compose_img_text_with_regions(imgs_query, modification_texts, scan_info)
        mod_img1 = self.normalization_layer(mod_img1)
        img2 = self.extract_img_feature(imgs_target)
        img2 = self.normalization_layer(img2)
        assert (mod_img1.shape[0] == img2.shape[0] and
                mod_img1.shape[1] == img2.shape[1])
        if soft_triplet_loss:
            return self.compute_soft_triplet_loss_(mod_img1, img2)
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

    def __init__(self, texts, embed_dim):
        super(ImgEncoderTextEncoderBase, self).__init__()

        # img model
        img_model = torchvision.models.resnet18(pretrained=True)

        class GlobalAvgPool2d(torch.nn.Module):

            def forward(self, x):
                return F.adaptive_avg_pool2d(x, (1, 1))

        img_model.avgpool = GlobalAvgPool2d()
        img_model.fc = torch.nn.Sequential(torch.nn.Linear(512, embed_dim))
        self.img_model = img_model

        # text model
        self.text_model = text_model.TextLSTMModel(
            texts_to_build_vocab=texts,
            word_embed_dim=embed_dim,
            lstm_hidden_dim=embed_dim)

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

    def __init__(self, texts, embed_dim):
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

    def __init__(self, texts, embed_dim):
        super(TIRG, self).__init__(texts, embed_dim)

        self.a = torch.nn.Parameter(torch.tensor([1.0, 10.0, 1.0, 1.0]))
        self.gated_feature_composer = torch.nn.Sequential(
            ConCatModule(), 
            torch.nn.BatchNorm1d(2 * embed_dim), 
            torch.nn.ReLU(),
            torch.nn.Linear(2 * embed_dim, embed_dim))
        self.res_info_composer = torch.nn.Sequential(
            ConCatModule(), 
            torch.nn.BatchNorm1d(2 * embed_dim), 
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
        f = F.sigmoid(f1) * img_features * self.a[0] + f2 * self.a[1]
        return f


# RNN Based Language Model
# class EncoderText(nn.Module):
#
#     def __init__(self, vocab_size, word_dim, embed_size, num_layers,
#                  use_bi_gru=False, no_txtnorm=False):
#         super(EncoderText, self).__init__()
#         self.embed_size = embed_size
#         self.no_txtnorm = no_txtnorm
#
#         # word embedding
#         self.embed = nn.Embedding(vocab_size, word_dim)
#
#         # caption embedding
#         self.use_bi_gru = use_bi_gru
#         self.rnn = nn.GRU(word_dim, embed_size, num_layers, batch_first=True, bidirectional=use_bi_gru)
#
#         self.init_weights()
#
#     def init_weights(self):
#         self.embed.weight.data.uniform_(-0.1, 0.1)
#
#     def forward(self, x, lengths):
#         """Handles variable size captions
#         """
#         # Embed word ids to vectors
#         x = self.embed(x)
#         packed = pack_padded_sequence(x, lengths, batch_first=True)
#
#         # Forward propagate RNN
#         out, _ = self.rnn(packed)
#
#         # Reshape *final* output to (batch_size, hidden_size)
#         padded = pad_packed_sequence(out, batch_first=True)
#         cap_emb, cap_len = padded
#
#         if self.use_bi_gru:
#             cap_emb = (cap_emb[:, :, :cap_emb.size(2) / 2] + cap_emb[:, :, cap_emb.size(2) / 2:]) / 2
#
#         # normalization in the joint embedding space
#         if not self.no_txtnorm:
#             cap_emb = l2norm(cap_emb, dim=-1)
#
#         return cap_emb, cap_len


def l2norm(X, dim, eps=1e-8):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X

class TIRGWithSCAN(ImgEncoderTextEncoderBase):
    """The TIGR model.

    The method is described in
    Nam Vo, Lu Jiang, Chen Sun, Kevin Murphy, Li-Jia Li, Li Fei-Fei, James Hays.
    "Composing Text and Image for Image Retrieval - An Empirical Odyssey"
    CVPR 2019. arXiv:1812.07119
    """

    def __init__(self, texts, embed_dim):
        super(TIRGWithSCAN, self).__init__(texts, embed_dim)

        import json
        self.id2index = json.load(open("../meta.json", "r"))
        self.img_enc = EncoderImage(2048, 512,
                                    precomp_enc_type='basic',
                                    no_imgnorm=True)
        if torch.cuda.is_available():
            self.img_enc.cuda()
        self.a = torch.nn.Parameter(torch.tensor([1.0, 10.0, 5.0, 1.0]))
        self.gated_feature_composer = torch.nn.Sequential(
            ConCatModule(), 
            torch.nn.BatchNorm1d(3 * embed_dim),
            torch.nn.Dropout(p=0.2),
            torch.nn.ReLU(),
            torch.nn.Linear(3 * embed_dim, embed_dim))
        self.res_info_composer = torch.nn.Sequential(
            ConCatModule(), 
            torch.nn.BatchNorm1d(3 * embed_dim), 
            torch.nn.Dropout(p=0.2),
            torch.nn.ReLU(),
            torch.nn.Linear(3 * embed_dim, 2 * embed_dim), 
            torch.nn.ReLU(),
            torch.nn.Linear(2 * embed_dim, embed_dim))
        
    def xattn_score_i2t_and_t2i(self, images, text_features, noun_features):
        """
        Images: (batch_size, n_regions, d) matrix of images
        Captions: (batch_size, max_n_words, d) matrix of captions
        CapLens: (batch_size) array of caption lengths
        """
        weiContext_t2i, _ = func_attention(torch.cat((text_features.unsqueeze(1), 
                                                      noun_features.unsqueeze(1)), 1),
                                                      images,
                                                      "softmax", smooth=9.)
        
        weiContext_i2t, _ = func_attention(images, torch.cat((text_features.unsqueeze(1), 
                                                      noun_features.unsqueeze(1)), 1),
                                                      "softmax", smooth=9.)

        return weiContext_t2i, weiContext_i2t

    #        return weiContext_i2t, weiContext_t2i
    #         for i in range(2):
    #             """
    #                 word(query): (n_image, n_word, d)
    #                 image(context): (n_image, n_region, d)
    #                 weiContext: (n_image, n_region, d)
    #                 attn: (n_image, n_word, n_region)
    #             """
    #             weiContext_i2t, attn_i2t = func_attention(images, captions, "softmax", smooth=9.)
    #             # weiContext_t2i, attn_t2i = func_attention(captions, images, "softmax", smooth=9.)
    #             print('i: ', i, 'weiContext_i2t', weiContext_i2t.shape)
    # return weiContext_i2t #, weiContext_t2i

    def extract_img_feature_with_attention(self, regions, text_features, noun_features):
        weiContext_t2i, weiContext_i2t = self.xattn_score_i2t_and_t2i(regions, text_features, noun_features)
        return weiContext_t2i, weiContext_i2t

    def compose_img_text_with_regions(self, imgs, texts, scan_info):
        img1_regions, nouns = scan_info
        
        text_features = self.extract_text_feature(texts)
        noun_features = self.extract_text_feature(nouns)
        
        # text_features = l2norm(text_features, dim=-1)
        # noun_features = l2norm(noun_features, dim=-1)

        img1_regions = torch.from_numpy(img1_regions).cuda()
        feats = self.img_enc(img1_regions)
        
        img_features = self.extract_img_feature(imgs)
        weiContext_t2i, weiContext_i2t = self.extract_img_feature_with_attention(feats,
                                                                 text_features,
                                                                 noun_features)

        self.text_context = torch.split(weiContext_t2i, 1, 1)
        self.image_context = torch.split(weiContext_i2t, 1, 1)
        
        self.image_context = [v.squeeze(1) for v in self.image_context]
        self.text_context = [v.squeeze(1) for v in self.text_context]
        # print(weiContext_i2t.shape, self.image_context[0].shape)
        
        # 32 x 3 x 512
#         print(self.image_context[0].shape, self.text_context[0].shape, len(self.text_context), 
#                                                                        len(self.image_context))
        
        # img_features_mult = torch.tensor(img_features) 
        text_features_mult = torch.tensor(text_features) 
        # for im_c in self.image_context:
            # print(img_features.shape, im_c.shape)
        #    img_features_mult += img_features * im_c.squeeze(1)
        for text_c in self.text_context:
            text_features_mult += text_features * text_c.squeeze(1)
            
        self.stacked = torch.cat([text_features_mult, 
                                  text_features], dim=1)
        
        # print(self.stacked.shape)
        # self.img_features_mult = l2norm(img_features_mult, dim=-1)
        # self.text_features_mult = l2norm(text_features_mult, dim=-1)
        
        return self.compose_img_text_features(img_features, text_features)

    def compose_img_text_features(self, img_features, text_features):

        f1 = self.gated_feature_composer((img_features, self.stacked))
        f2 = self.res_info_composer((img_features, self.stacked))
        
        # f3 = self.res_info_composer((self.img_features_mult, self.text_features_mult))
        f = F.sigmoid(f1) * img_features * self.a[0] + f2 * self.a[1] # + f3 * img_features * self.a[2] 
        return f


def l1norm(X, dim, eps=1e-8):
    """L1-normalize columns of X
    """
    norm = torch.abs(X).sum(dim=dim, keepdim=True) + eps
    X = torch.div(X, norm)
    return X


def l2norm(X, dim, eps=1e-8):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X


def EncoderImage(img_dim, embed_size, precomp_enc_type='basic',
                 no_imgnorm=False):
    """A wrapper to image encoders. Chooses between an different encoders
    that uses precomputed image features.
    """
    if precomp_enc_type == 'basic':
        img_enc = EncoderImagePrecomp(
            img_dim, embed_size, no_imgnorm)
    elif precomp_enc_type == 'weight_norm':
        img_enc = EncoderImageWeightNormPrecomp(
            img_dim, embed_size, no_imgnorm)
    else:
        raise ValueError("Unknown precomp_enc_type: {}".format(precomp_enc_type))

    return img_enc


class EncoderImagePrecomp(nn.Module):

    def __init__(self, img_dim, embed_size, no_imgnorm=False):
        super(EncoderImagePrecomp, self).__init__()
        self.embed_size = embed_size
        self.no_imgnorm = no_imgnorm
        self.fc = nn.Linear(img_dim, embed_size)

        self.init_weights()

    def init_weights(self):
        """Xavier initialization for the fully connected layer
        """
        r = np.sqrt(6.) / np.sqrt(self.fc.in_features +
                                  self.fc.out_features)
        self.fc.weight.data.uniform_(-r, r)
        self.fc.bias.data.fill_(0)

    def forward(self, images):
        """Extract image feature vectors."""
        # assuming that the precomputed features are already l2-normalized

        features = self.fc(images)

        # normalize in the joint embedding space
        if not self.no_imgnorm:
            features = l2norm(features, dim=-1)

        return features

    def load_state_dict(self, state_dict):
        """Copies parameters. overwritting the default one to
        accept state_dict from Full model
        """
        own_state = self.state_dict()
        new_state = OrderedDict()
        for name, param in state_dict.items():
            if name in own_state:
                new_state[name] = param

        super(EncoderImagePrecomp, self).load_state_dict(new_state)


class EncoderImageWeightNormPrecomp(nn.Module):

    def __init__(self, img_dim, embed_size, no_imgnorm=False):
        super(EncoderImageWeightNormPrecomp, self).__init__()
        self.embed_size = embed_size
        self.no_imgnorm = no_imgnorm
        self.fc = weight_norm(nn.Linear(img_dim, embed_size), dim=None)

    def forward(self, images):
        """Extract image feature vectors."""
        # assuming that the precomputed features are already l2-normalized

        features = self.fc(images)

        # normalize in the joint embedding space
        if not self.no_imgnorm:
            features = l2norm(features, dim=-1)

        return features

    def load_state_dict(self, state_dict):
        """Copies parameters. overwritting the default one to
        accept state_dict from Full model
        """
        own_state = self.state_dict()
        new_state = OrderedDict()
        for name, param in state_dict.items():
            if name in own_state:
                new_state[name] = param

        super(EncoderImageWeightNormPrecomp, self).load_state_dict(new_state)


class TIRGLastConv(ImgEncoderTextEncoderBase):
    """The TIGR model with spatial modification over the last conv layer.

    The method is described in
    Nam Vo, Lu Jiang, Chen Sun, Kevin Murphy, Li-Jia Li, Li Fei-Fei, James Hays.
    "Composing Text and Image for Image Retrieval - An Empirical Odyssey"
    CVPR 2019. arXiv:1812.07119
    """

    def __init__(self, texts, embed_dim):
        super(TIRGLastConv, self).__init__(texts, embed_dim)

        self.a = torch.nn.Parameter(torch.tensor([1.0, 10.0, 1.0, 1.0]))
        self.mod2d = torch.nn.Sequential(
            torch.nn.BatchNorm2d(512 + embed_dim),
            torch.nn.Conv2d(512 + embed_dim, 512 + embed_dim, [3, 3], padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(512 + embed_dim, 512, [3, 3], padding=1),
        )

        self.mod2d_gate = torch.nn.Sequential(
            torch.nn.BatchNorm2d(512 + embed_dim),
            torch.nn.Conv2d(512 + embed_dim, 512 + embed_dim, [3, 3], padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(512 + embed_dim, 512, [3, 3], padding=1),
        )

    def compose_img_text(self, imgs, texts):
        text_features = self.extract_text_feature(texts)

        x = imgs
        x = self.img_model.conv1(x)
        x = self.img_model.bn1(x)
        x = self.img_model.relu(x)
        x = self.img_model.maxpool(x)

        x = self.img_model.layer1(x)
        x = self.img_model.layer2(x)
        x = self.img_model.layer3(x)
        x = self.img_model.layer4(x)

        # mod
        y = text_features
        y = y.reshape((y.shape[0], y.shape[1], 1, 1)).repeat(
            1, 1, x.shape[2], x.shape[3])
        z = torch.cat((x, y), dim=1)
        t = self.mod2d(z)
        tgate = self.mod2d_gate(z)
        x = self.a[0] * F.sigmoid(tgate) * x + self.a[1] * t

        x = self.img_model.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.img_model.fc(x)
        return x


def func_attention(query, context, raw_feature_norm, smooth, eps=1e-8):
    """
    query: (n_context, queryL, d)
    context: (n_context, sourceL, d)
    """
    batch_size_q, queryL = query.size(0), query.size(1)
    batch_size, sourceL = context.size(0), context.size(1)

    # Get attention
    # --> (batch, d, queryL)
    queryT = torch.transpose(query, 1, 2)

    # (batch, sourceL, d)(batch, d, queryL)
    # --> (batch, sourceL, queryL)
    attn = torch.bmm(context, queryT)
    if raw_feature_norm == "softmax":
        # --> (batch*sourceL, queryL)
        attn = attn.view(batch_size * sourceL, queryL)
        attn = torch.nn.Softmax()(attn)
        # --> (batch, sourceL, queryL)
        attn = attn.view(batch_size, sourceL, queryL)
    elif raw_feature_norm == "l2norm":
        attn = l2norm(attn, 2)
    elif raw_feature_norm == "clipped_l2norm":
        attn = nn.LeakyReLU(0.1)(attn)
        attn = l2norm(attn, 2)
    elif raw_feature_norm == "l1norm":
        attn = l1norm(attn, 2)
    elif raw_feature_norm == "clipped_l1norm":
        attn = torch.nn.LeakyReLU(0.1)(attn)
        attn = l1norm(attn, 2)
    elif raw_feature_norm == "clipped":
        attn = torch.nn.LeakyReLU(0.1)(attn)
    elif raw_feature_norm == "no_norm":
        pass
    else:
        raise ValueError("unknown first norm type:", raw_feature_norm)
    # --> (batch, queryL, sourceL)
    attn = torch.transpose(attn, 1, 2).contiguous()
    # --> (batch*queryL, sourceL)
    attn = attn.view(batch_size * queryL, sourceL)
    attn = torch.nn.Softmax()(attn * smooth)
    # --> (batch, queryL, sourceL)
    attn = attn.view(batch_size, queryL, sourceL)
    # --> (batch, sourceL, queryL)
    attnT = torch.transpose(attn, 1, 2).contiguous()

    # --> (batch, d, sourceL)
    contextT = torch.transpose(context, 1, 2)
    # (batch x d x sourceL)(batch x sourceL x queryL)
    # --> (batch, d, queryL)
    weightedContext = torch.bmm(contextT, attnT)
    # --> (batch, queryL, d)
    weightedContext = torch.transpose(weightedContext, 1, 2)

    return weightedContext, attnT
