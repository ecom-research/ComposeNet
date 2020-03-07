
# TODO(lujiang): put it into the third-party
# MIT License

# Copyright (c) 2018 Nam Vo

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

"""Metric learning functions.

Codes are modified from:
https://github.com/lugiavn/generalization-dml/blob/master/nams.py
"""

import numpy as np
import torch
import torchvision
from torch.functional import F


def pairwise_distances(x, y=None):
  """
    Input: x is a Nxd matrix
           y is an optional Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between
    x[i,:] and y[j,:]
            if y is not given then use 'y=x'.
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    source:
    https://discuss.pytorch.org/t/efficient-distance-matrix-computation/9065/2
    """
  x_norm = (x**2).sum(1).view(-1, 1)
  if y is not None:
    y_t = torch.transpose(y, 0, 1)
    y_norm = (y**2).sum(1).view(1, -1)
  else:
    y_t = torch.transpose(x, 0, 1)
    y_norm = x_norm.view(1, -1)

  dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
  # Ensure diagonal is zero if x=y
  # if y is None:
  #     dist = dist - torch.diag(dist.diag)
  return torch.clamp(dist, 0.0, np.inf)


def cos_distances(x):
    
    return F.cosine_similarity(x, y, dim=0)


class MyTripletLossFunc(torch.autograd.Function):

  def __init__(self, triplets):
    super(MyTripletLossFunc, self).__init__()
    self.triplets = triplets
    self.triplet_count = len(triplets)

  def forward(self, image_features, text_features):
    self.save_for_backward(image_features, text_features)

    self.distances_image = pairwise_distances(image_features).cpu().numpy()
    self.distances_text = pairwise_distances(text_features).cpu().numpy()
    
    # print(image_features.shape, text_features.shape, self.distances_image.shape, self.distances_text.shape)

    loss_image = 0.0
    loss_text = 0.0
    triplet_count = 0.0
    correct_count = 0.0
    for i, j, k in self.triplets:
      w = 1.0
      triplet_count += w
      loss_image += w * np.log(1 + np.exp(self.distances_image[i, j] - self.distances_image[i, k]))
      loss_text += w * np.log(1 + np.exp(self.distances_text[i, j] - self.distances_text[i, k]))
      if self.distances_image[i, j] < self.distances_image[i, k]:
        correct_count += 1

    loss_image /= triplet_count
    loss_text /= triplet_count
    # loss /= triplet_count
    return torch.FloatTensor((loss_image,)), torch.FloatTensor((loss_text,))

  def backward(self, grad_image_output, grad_text_output):
    image_features, text_features = self.saved_tensors
    
    image_features_np = image_features.cpu().numpy()
    text_features_np = text_features.cpu().numpy()
    
    grad_image_features = image_features.clone() * 0.0
    grad_image_features_np = grad_image_features.cpu().numpy()
    
    grad_text_features = text_features.clone() * 0.0
    grad_text_features_np = grad_text_features.cpu().numpy()

    for i, j, k in self.triplets:
      w = 1.0
      f_img = 1.0 - 1.0 / (
          1.0 + np.exp(self.distances_image[i, j] - self.distances_image[i, k]))
      f_txt = 1.0 - 1.0 / (
          1.0 + np.exp(self.distances_text[i, j] - self.distances_text[i, k]))
        
      grad_image_features_np[i, :] += w * f_img * (
          image_features_np[i, :] - image_features_np[j, :]) / self.triplet_count
      grad_image_features_np[j, :] += w * f_img * (
          image_features_np[j, :] - image_features_np[i, :]) / self.triplet_count
      grad_image_features_np[i, :] += -w * f_img * (
          image_features_np[i, :] - image_features_np[k, :]) / self.triplet_count
      grad_image_features_np[k, :] += -w * f_img * (
          image_features_np[k, :] - image_features_np[i, :]) / self.triplet_count
        
      grad_text_features_np[i, :] += w * f_txt * (
          text_features_np[i, :] - text_features_np[j, :]) / self.triplet_count
      grad_text_features_np[j, :] += w * f_txt * (
          text_features_np[j, :] - text_features_np[i, :]) / self.triplet_count
      grad_text_features_np[i, :] += -w * f_txt * (
          text_features_np[i, :] - text_features_np[k, :]) / self.triplet_count
      grad_text_features_np[k, :] += -w * f_txt * (
          text_features_np[k, :] - text_features_np[i, :]) / self.triplet_count

    for i in range(image_features_np.shape[0]):
      grad_image_features[i, :] = torch.from_numpy(grad_image_features_np[i, :])
      grad_text_features[i, :] = torch.from_numpy(grad_text_features_np[i, :])
    grad_image_features *= float(grad_image_output.data[0])
    grad_text_features *= float(grad_text_output.data[0])
    
    return grad_image_features, grad_text_features


class TripletLoss(torch.nn.Module):
  """Class for the triplet loss."""
  def __init__(self, pre_layer=None):
    super(TripletLoss, self).__init__()
    self.pre_layer = pre_layer
    self.norm_layer = NormalizationLayer()

  def forward(self, x, y, triplets):
    if self.pre_layer is not None:
      x = self.pre_layer(x)
    y = self.norm_layer(y)
    loss_im, loss_txt = MyTripletLossFunc(triplets)(x, y)
    # print("current_losses loss_im, loss_txt", loss_im, loss_txt, "\n")
    
    return loss_im + loss_txt


class NormalizationLayer(torch.nn.Module):
  """Class for normalization layer."""
  def __init__(self, normalize_scale=1.0, learn_scale=True):
    super(NormalizationLayer, self).__init__()
    self.norm_s = float(normalize_scale)
    if learn_scale:
      self.norm_s = torch.nn.Parameter(torch.FloatTensor((self.norm_s,)))

  def forward(self, x):
    features = self.norm_s * x / torch.norm(x, dim=1, keepdim=True).expand_as(x)
    return features
