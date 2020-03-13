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

class TextModule(torch.nn.Module):

    def __init__(self):
        super(TextModule, self).__init__()
        self.bert_features = torch.nn.Sequential(
            torch.nn.BatchNorm1d(768),
            torch.nn.Linear(768, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 512)
        )
        
    def forward(self, x):
        return self.bert_features(x)
    

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
        self.wv = KeyedVectors.load('../tirg-with-scan/wordvectors-300.kv', mmap='r')
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
        
#         self.text_processing = torch.nn.Sequential(
#             torch.nn.BatchNorm1d(300),
#             torch.nn.Linear(300, 300),
#             torch.nn.ReLU(),
#             torch.nn.Linear(300, 300)
#         )
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

    def compose_img_text_with_extra_data(self, imgs, texts, extra_data):
        batch_size = imgs.shape[0]
        img_features = self.extract_img_feature(imgs)
        
        # region descriptions
#         source_context_features = [' '.join(x.values()) for x in extra_data[1]]
#         target_context_features = [' '.join(x.values()) for x in extra_data[2]]
        source_adjs = [x.split(' ')[0] for x in extra_data[3]]
        target_adjs = texts
        nouns = extra_data[0]
        
        # self.arithmetic_repr, self.target_repr = self.get_w2v_representation(source_adjs, target_adjs, nouns, batch_size)
        
        # adding nouns to mods, target representation
        text_features = [adj + " " + noun for adj, noun in zip(texts, extra_data[0])]
        text_features = torch.from_numpy(bc.encode(text_features)).cuda()
        
#         joined = text_features + extra_data[3]
#         joined = bc.encode(joined)
        
#         text_features = torch.from_numpy(joined[:batch_size]).cuda()
#         source_text_features = torch.from_numpy(joined[batch_size:]).cuda()
        
#         self.source_text_feats = source_text_features
#         self.target_text_feats = text_features
        
        # text_features_lstm = self.extract_text_feature(texts)
        
#         self.source_text_feats = self.text_processing(source_text_features)
#         self.target_text_feats = self.text_processing(text_features)
        
#         joined = source_context_features + target_context_features + text_features
#         joined = bc.encode(joined)
        
#         source_context_features = torch.from_numpy(joined[:batch_size]).cuda()
#         target_context_features = torch.from_numpy(joined[batch_size:2*batch_size]).cuda()
#         text_features = torch.from_numpy(joined[2*batch_size:]).cuda()
        
#         self.source_text_feats = self.text_processing(source_context_features)
#         self.target_text_feats = self.text_processing(target_context_features)

        return self.compose_img_text_features(img_features, text_features)

    def compose_img_text_features(self, img_features, text_features):
#         arithmetic_repr = self.text_processing(self.arithmetic_repr)
#         target_repr = self.text_processing(self.target_repr)

        # "target" text, just like img2
        mapped_text_features_original = self.text_features_mapping(text_features)
        mapped_image_features_original = self.image_features_mapping(img_features)
        
        f1_image = self.image_gated_feature_composer((mapped_image_features_original,
                                                      mapped_text_features_original))
        f2_image = self.image_res_info_composer((mapped_image_features_original, 
                                                 mapped_text_features_original))
        
        f_image = torch.sigmoid(f1_image) * img_features * self.a[0] + self.a[1] * f2_image
        
        
        
        
        f1_text = self.text_gated_feature_composer((mapped_image_features_original,
                                                    mapped_text_features_original))
        f2_text = self.text_res_info_composer((mapped_image_features_original, 
                                               mapped_text_features_original))
        
        f_text = torch.sigmoid(f1_text) * text_features * self.b[0] + self.b[1] * f2_text
        
        return f_image, f_text, text_features
    
    
'''
An alternative implementation for PyTorch with auto-infering the x-y dimensions.
'''
class AddCoords(nn.Module):

    def __init__(self, with_r=False):
        super(AddCoords, self).__init__()
        self.with_r = with_r

    def forward(self, input_tensor):
        """
        Args:
            input_tensor: shape(batch, channel, x_dim, y_dim)
        """
        batch_size, _, x_dim, y_dim = input_tensor.size()

        xx_channel = torch.arange(x_dim).repeat(1, y_dim, 1)
        yy_channel = torch.arange(y_dim).repeat(1, x_dim, 1).transpose(1, 2)

        xx_channel = xx_channel.float() / (x_dim - 1)
        yy_channel = yy_channel.float() / (y_dim - 1)

        xx_channel = xx_channel * 2 - 1
        yy_channel = yy_channel * 2 - 1

        xx_channel = xx_channel.repeat(batch_size, 1, 1, 1).transpose(2, 3)
        yy_channel = yy_channel.repeat(batch_size, 1, 1, 1).transpose(2, 3)

        ret = torch.cat([
            input_tensor,
            xx_channel.type_as(input_tensor),
            yy_channel.type_as(input_tensor)], dim=1)

        if self.with_r:
            rr = torch.sqrt(torch.pow(xx_channel.type_as(input_tensor) - 0.5, 2) + torch.pow(yy_channel.type_as(input_tensor) - 0.5, 2))
            ret = torch.cat([ret, rr], dim=1)

        return ret


class CoordConv(nn.Module):

    def __init__(self, in_channels, out_channels, with_r=False, kernel_size=5, stride=3, padding=2):
        super(CoordConv, self).__init__()
        self.addcoords = AddCoords(with_r=with_r)
        in_size = in_channels+2
        if with_r:
            in_size += 1
        self.conv = torch.nn.Conv2d(in_size, out_channels, kernel_size=5, stride=3, padding=2)

    def forward(self, x):
        ret = self.addcoords(x)
        ret = self.conv(ret)
        return ret
    
    
class Attention(torch.nn.Module):
    """ Applies attention mechanism on the `context` using the `query`.

    **Thank you** to IBM for their initial implementation of :class:`Attention`. Here is
    their `License
    <https://github.com/IBM/pytorch-seq2seq/blob/master/LICENSE>`__.

    Args:
        dimensions (int): Dimensionality of the query and context.
        attention_type (str, optional): How to compute the attention score:

            * dot: :math:`score(H_j,q) = H_j^T q`
            * general: :math:`score(H_j, q) = H_j^T W_a q`

    Example:

         >>> attention = Attention(256)
         >>> query = torch.randn(5, 1, 256)
         >>> context = torch.randn(5, 5, 256)
         >>> output, weights = attention(query, context)
         >>> output.size()
         torch.Size([5, 1, 256])
         >>> weights.size()
         torch.Size([5, 1, 5])
    """

    def __init__(self, dimensions, attention_type='general'):
        super(Attention, self).__init__()

        if attention_type not in ['dot', 'general']:
            raise ValueError('Invalid attention type selected.')

        self.attention_type = attention_type
        if self.attention_type == 'general':
            self.linear_in = torch.nn.Linear(dimensions, dimensions, bias=False).to('cuda')

        self.linear_out = torch.nn.Linear(dimensions * 2, dimensions, bias=False).to('cuda')

    def forward(self, query, context):
        """
        Args:
            query (:class:`torch.FloatTensor` [batch size, output length, dimensions]): Sequence of
                queries to query the context.
            context (:class:`torch.FloatTensor` [batch size, query length, dimensions]): Data
                overwhich to apply the attention mechanism.

        Returns:
            :class:`tuple` with `output` and `weights`:
            * **output** (:class:`torch.LongTensor` [batch size, output length, dimensions]):
              Tensor containing the attended features.
            * **weights** (:class:`torch.FloatTensor` [batch size, output length, query length]):
              Tensor containing attention weights.
        """
        batch_size, output_len, dimensions = query.size()
        query_len = context.size(1)

        if self.attention_type == "general":
            query = query.reshape(batch_size * output_len, dimensions)
            query = self.linear_in(query)
            query = query.reshape(batch_size, output_len, dimensions)

        # TODO: Include mask on PADDING_INDEX?

        # (batch_size, output_len, dimensions) * (batch_size, query_len, dimensions) ->
        # (batch_size, output_len, query_len)
        attention_scores = torch.bmm(query, context.transpose(1, 2).contiguous())

        # Compute weights across every context sequence
        attention_scores = attention_scores.view(batch_size * output_len, query_len)
        attention_weights = torch.nn.functional.softmax(attention_scores, dim=-1)
        attention_weights = attention_weights.view(batch_size, output_len, query_len)

        # (batch_size, output_len, query_len) * (batch_size, query_len, dimensions) ->
        # (batch_size, output_len, dimensions)
        mix = torch.bmm(attention_weights, context)

        # concat -> (batch_size * output_len, 2*dimensions)
        combined = torch.cat((mix, query), dim=2)
        combined = combined.view(batch_size * output_len, 2 * dimensions)

        # Apply linear_out on every 2nd dimension of concat
        # output -> (batch_size, output_len, dimensions)
        output = self.linear_out(combined).view(batch_size, output_len, dimensions)
        output = torch.tanh(output)

        return output, attention_weights



def func_attention(query, context, raw_feature_norm, smooth=9., eps=1e-8):
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
        attn = attn.view(batch_size*sourceL, queryL)
        attn = torch.nn.functional.softmax(attn)
        # --> (batch, sourceL, queryL)
        attn = attn.view(batch_size, sourceL, queryL)
    elif raw_feature_norm == "l2norm":
        attn = l2norm(attn, 2)
    elif raw_feature_norm == "clipped_l2norm":
        attn = torch.nn.functional.leaky_relu(attn, 0.1)
        attn = l2norm(attn, 2)
    elif raw_feature_norm == "l1norm":
        attn = l1norm_d(attn, 2)
    elif raw_feature_norm == "clipped_l1norm":
        attn = torch.nn.functional.leaky_relu(attn, 0.1)
        attn = l1norm_d(attn, 2)
    elif raw_feature_norm == "clipped":
        attn = torch.nn.functional.leaky_relu(attn, 0.1)
    elif raw_feature_norm == "no_norm":
        pass
    else:
        raise ValueError("unknown first norm type:", opt.raw_feature_norm)
    # --> (batch, queryL, sourceL)
    attn = torch.transpose(attn, 1, 2).contiguous()
    # --> (batch*queryL, sourceL)
    attn = attn.view(batch_size*queryL, sourceL)
    attn = torch.nn.functional.softmax(attn*smooth)
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