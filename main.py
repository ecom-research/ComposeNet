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

"""Main method to train the model."""


#!/usr/bin/python

import argparse
import sys
import time
import ast
import gc
import datasets
import img_text_composition_models
import numpy as np
from tensorboardX import SummaryWriter
from torch.autograd import Variable
import test_retrieval
import torch
import torch.utils.data
import torchvision
from tqdm import tqdm as tqdm
from copy import deepcopy

torch.set_num_threads(3)


def parse_opt():
  """Parses the input arguments."""
  parser = argparse.ArgumentParser()
  parser.add_argument('-f', type=str, default='')
  parser.add_argument('--comment', type=str, default='test_notebook')
  parser.add_argument('--dataset', type=str, default='css3d')
  parser.add_argument(
      '--dataset_path', type=str, default='../imgcomsearch/CSSDataset/output')
  parser.add_argument('--model', type=str, default='tirg')
  parser.add_argument('--embed_dim', type=int, default=512)
  parser.add_argument('--learning_rate', type=float, default=1e-2)
  parser.add_argument(
      '--learning_rate_decay_frequency', type=int, default=9999999)
  parser.add_argument('--batch_size', type=int, default=32)
  parser.add_argument('--weight_decay', type=float, default=1e-6)
  parser.add_argument('--num_iters', type=int, default=210000)
  parser.add_argument('--loss', type=str, default='soft_triplet')
  parser.add_argument('--loader_num_workers', type=int, default=4)
  parser.add_argument('--log_dir', type=str, default='../logs/')
  parser.add_argument('--test_only', type=bool, default=False)
  parser.add_argument('--model_checkpoint', type=str, default='')
  parser.add_argument('--learn_on_regions', type=bool, default=False)
  parser.add_argument('--use_pretrained', type=bool, default=False)
  parser.add_argument('--optimizer', type=str, default='SGD')
  args = parser.parse_args()
  return args


def switch_weights(full_model, regions_model):
    new_model = deepcopy(full_model)
    for regions_model_field, tirg_model_field in zip(regions_model['model_state_dict'].items(), 
                                                     full_model.items()):
        current_field = regions_model_field[0]
        if current_field in full_model.keys() and 'text_model' not in current_field:
            new_model[current_field] = \
            regions_model['model_state_dict'][current_field]

    
    return new_model

def load_dataset(opt):
  """Loads the input datasets."""
  print 'Reading dataset ', opt.dataset
  if opt.dataset == 'css3d':
    trainset = datasets.CSSDataset(
        path=opt.dataset_path,
        split='train',
        transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                             [0.229, 0.224, 0.225])
        ]))
    testset = datasets.CSSDataset(
        path=opt.dataset_path,
        split='test',
        transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                             [0.229, 0.224, 0.225])
        ]))
  elif opt.dataset == 'fashion200k':
    trainset = datasets.Fashion200k(
        path=opt.dataset_path,
        split='train',
        transform=torchvision.transforms.Compose([
            torchvision.transforms.Resize(224),
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                             [0.229, 0.224, 0.225])
        ]))
    testset = datasets.Fashion200k(
        path=opt.dataset_path,
        split='test',
        transform=torchvision.transforms.Compose([
            torchvision.transforms.Resize(224),
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                             [0.229, 0.224, 0.225])
        ]))
  elif opt.dataset == 'mitstates':
    trainset = datasets.MITStates(
        path=opt.dataset_path,
        split='train',
        transform=torchvision.transforms.Compose([
            torchvision.transforms.Resize(224),
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                             [0.229, 0.224, 0.225])
        ]))
    testset = datasets.MITStates(
        path=opt.dataset_path,
        split='test',
        transform=torchvision.transforms.Compose([
            torchvision.transforms.Resize(224),
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                             [0.229, 0.224, 0.225])
        ]))
  elif opt.dataset == 'mitstates_AND_regions':
    trainset = datasets.MITStatesANDRegions(
        path=opt.dataset_path,
        split='train',
        transform=torchvision.transforms.Compose([
            torchvision.transforms.Resize(224),
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                             [0.229, 0.224, 0.225])
        ]))
    testset = datasets.MITStatesANDRegions(
        path=opt.dataset_path,
        split='test',
        transform=torchvision.transforms.Compose([
            torchvision.transforms.Resize(224),
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                             [0.229, 0.224, 0.225])
        ]))
    
  elif opt.dataset == 'mitstates_regions':
    trainset = datasets.MITStatesRegions(
        path=opt.dataset_path, # '../regions_info_data.txt'
        split='train')
    testset = datasets.MITStatesRegions(
        path=opt.dataset_path,
        split='test')
  else:
    print 'Invalid dataset', opt.dataset
    sys.exit()

  print 'trainset size:', len(trainset)
  print 'testset size:', len(testset)
  return trainset, testset


def create_model_and_optimizer(opt, texts):
  """Builds the model and related optimizer."""
  print 'Creating model and optimizer for', opt.model
  if opt.model == 'imgonly':
    model = img_text_composition_models.SimpleModelImageOnly(
        texts, embed_dim=opt.embed_dim, learn_on_regions=opt.learn_on_regions)
  elif opt.model == 'textonly':
    model = img_text_composition_models.SimpleModelTextOnly(
        texts, embed_dim=opt.embed_dim, learn_on_regions=opt.learn_on_regions)
  elif opt.model == 'concat':
    model = img_text_composition_models.Concat(texts, embed_dim=opt.embed_dim)
  elif opt.model == 'tirg':
    model = img_text_composition_models.TIRG(
        texts, embed_dim=opt.embed_dim, learn_on_regions=opt.learn_on_regions)
    if opt.use_pretrained:
        # print("Using regions pretrained model from ", opt.model_checkpoint)
        
        regions_model_checkpoint = torch.load('../logs/mitstates/Feb22_21-33-49_ip-172-31-38-215mitstates_tirg_evolved_regions_05Drop/latest_checkpoint.pth')
        full_model_checkpoint = torch.load('../logs/mitstates/Feb15_22-52-59_ip-172-31-38-215mitstates_tirg_original_text_model/latest_checkpoint.pth')
        print("Switching weights...")
        model_state = switch_weights(full_model_checkpoint, regions_model_checkpoint)
        model.load_state_dict(model_state['model_state_dict'])
        if not opt.test_only:
            print("Preparing to continue training...")
            model.train()
            
  elif opt.model == 'tirg_lastconv':
    model = img_text_composition_models.TIRGLastConv(
        texts, embed_dim=opt.embed_dim, learn_on_regions=opt.learn_on_regions)
  elif opt.model == 'tirg_evolved':
    model = img_text_composition_models.TIRGEvolved(
        texts, embed_dim=opt.embed_dim, learn_on_regions=opt.learn_on_regions)
    if opt.use_pretrained:
        model_checkpoint = torch.load(opt.model_checkpoint)
        model_state = switch_weights(model.state_dict(), model_checkpoint)
        model.load_state_dict(model_state)
        print("Switched weights...")
        if not opt.test_only:
            print("Preparing to continue training...")
            model.train()
  elif opt.model == 'tirg_lastconv_evolved':
    model = img_text_composition_models.TIRGLastConvEvolved(
        texts, embed_dim=opt.embed_dim, learn_on_regions=opt.learn_on_regions)
  else:
    print 'Invalid model', opt.model
    print 'available: imgonly, textonly, concat, tirg, tirg_lastconv or tirg_evolved'
    sys.exit()
  model = model.cuda()

  # create optimizer
  params = []
  # low learning rate for pretrained layers on real image datasets
  if opt.dataset != 'css3d':
    params.append({
        'params': [p for p in model.img_model.fc.parameters()],
        'lr': opt.learning_rate
    })
    params.append({
        'params': [p for p in model.img_model.parameters()],
        'lr': 0.1 * opt.learning_rate
    })
  params.append({'params': [p for p in model.parameters()]})
  for _, p1 in enumerate(params):  # remove duplicated params
    for _, p2 in enumerate(params):
      if p1 is not p2:
        for p11 in p1['params']:
          for j, p22 in enumerate(p2['params']):
            if p11 is p22:
              p2['params'][j] = torch.tensor(0.0, requires_grad=True)
            
#   lr = 0.0001
#   b1 = 0.5
#   b2 = 0.999
  if opt.optimizer != 'SGD':
    optimizer = torch.optim.Adam(params, lr=opt.learning_rate, weight_decay=opt.weight_decay)
  else:
    optimizer = torch.optim.SGD(
      params, lr=opt.learning_rate, 
              momentum=0.9, 
              weight_decay=opt.weight_decay
  )
  
  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=10, eta_min=1e-5)

  return model, optimizer, scheduler


class Discriminator(torch.nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = torch.nn.Sequential(
            torch.nn.Linear(512, 64),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Linear(64, 1),
            torch.nn.Sigmoid(),
        )

    def forward(self, z):
        validity = self.model(z)
        return validity


def train_loop(opt, logger, trainset, testset, model, optimizer, scheduler):
  """Function for train loop"""
  print 'Begin training'
  torch.backends.cudnn.benchmark = True
  losses_tracking = {}
  it = 0
  epoch = -1
  tic = time.time()

  lr = 0.001
  b1 = 0.5
  b2 = 0.999
  # discriminator = Discriminator().cuda()
  # adversarial_loss = torch.nn.BCELoss().cuda()
  # optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(b1, b2))
  # optimizer_D = torch.optim.SGD(discriminator.parameters(), lr=lr, weight_decay=opt.weight_decay)
  cosine_loss = torch.nn.CosineEmbeddingLoss(margin=0.5).cuda()
  cosine_losstxt = torch.nn.CosineEmbeddingLoss(margin=0.5).cuda()
  while it < opt.num_iters:
    epoch += 1

    # show/log stats
    print 'It', it, 'epoch', epoch, 'Elapsed time', round(time.time() - tic,
                                                          4), opt.comment
    tic = time.time()
    for loss_name in losses_tracking:
      avg_loss = np.mean(losses_tracking[loss_name][-len(trainloader):])
      print '    Loss', loss_name, round(avg_loss, 4)
      logger.add_scalar(loss_name, avg_loss, it)
    logger.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], it)
    
    if epoch % 1 == 0:
      gc.collect()

    # test
    if epoch % 3 == 1:
      gc.collect()
      torch.cuda.empty_cache()
      tests = []
      for name, dataset in [('train', trainset), ('test', testset)]:
        t = test_retrieval.test(opt, model, dataset)
        tests += [(name + ' ' + metric_name, metric_value)
                  for metric_name, metric_value in t]
      for metric_name, metric_value in tests:
        logger.add_scalar(metric_name, metric_value, it)
        print '    ', metric_name, round(metric_value, 4)

    # save checkpoint
    torch.save({
        'it': it,
        'opt': opt,
        'model_state_dict': model.state_dict(),
    },
               logger.file_writer.get_logdir() + '/latest_checkpoint.pth')
    # return
    # run trainning for 1 epoch
    model.train()
    trainloader = trainset.get_loader(
        batch_size=opt.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=opt.loader_num_workers)
    
    hasOptimized = False
    valid = Variable(torch.cuda.FloatTensor(opt.batch_size, 1).fill_(1.0), requires_grad=False)
    fake = Variable(torch.cuda.FloatTensor(opt.batch_size, 1).fill_(0.0), requires_grad=False)
    fake_minus = Variable(torch.cuda.FloatTensor(opt.batch_size, 1).fill_(-1.0), requires_grad=False)
    def training_1_iter(data):
      assert type(data) is list
      valid = Variable(torch.cuda.FloatTensor(opt.batch_size, 1).fill_(1.0), requires_grad=False)
      fake = Variable(torch.cuda.FloatTensor(opt.batch_size, 1).fill_(0.0), requires_grad=False)
      fake_minus = Variable(torch.cuda.FloatTensor(opt.batch_size, 1).fill_(-1.0), requires_grad=False)
      img1 = np.stack([d['source_img_data'] for d in data])
      img1 = torch.from_numpy(img1).float()
      img1 = torch.autograd.Variable(img1).cuda()
      if opt.dataset in ['mitstates', 'mitstates_regions', 'mscoco_regions'] and opt.model == 'tirg_evolved':
          extra_data = [str(d['noun']) for d in data]
      elif opt.dataset == 'fashion200k' and opt.model == 'tirg_evolved':
          extra_data = [str(d['target_caption']) for d in data]

      img2 = np.stack([d['target_img_data'] for d in data])
      img2 = torch.from_numpy(img2).float()
      img2 = torch.autograd.Variable(img2).cuda()
      mods = [str(d['mod']['str']) for d in data]
      mods = [t.decode('utf-8') for t in mods]
      
      if opt.dataset == 'css3d':
          objects = [d['source_img_objects'] for d in data]
          descr_data = []
          for obj in objects:
              obj_desc = []
              for desc in obj:
                  if desc['shape']:
                      obj_desc.append(desc['pos_str'] + ' ' + desc['size'] + ' ' + desc['color'] + ' ' + desc['shape'])
              descr_data.append(obj_desc)
            
          target_objects = [d['target_img_objects'] for d in data]
          target_descr_data = []
          for obj in target_objects:
              obj_desc = []
              for desc in obj:
                  if desc['shape']:
                      obj_desc.append(desc['pos_str'] + ' ' + desc['size'] + ' ' + desc['color'] + ' ' + desc['shape'])
              target_descr_data.append(obj_desc)
          extra_data = [target_descr_data, descr_data]


      # compute loss
      losses = []
      # original tirg
      if opt.loss == 'soft_triplet' and opt.model not in ['tirg_evolved', 'tirg_lastconv_evolved']:
        loss_value = model.compute_loss(img1, 
                                        mods, 
                                        img2, 
                                        soft_triplet_loss=True)
      # experiments with lastconv
      elif opt.model == 'tirg_lastconv_evolved':
        loss_value = model.compute_loss_with_extra_data(img1, 
                                                        mods, 
                                                        img2, 
                                                        mods, # hard_coded 
                                                        soft_triplet_loss=True)
        
      # tirg evolved
      elif opt.loss == 'soft_triplet' and opt.model == 'tirg_evolved':
        loss_value, img2, encoded_imgs, img1, repr_to_compare_with_source, repr_to_compare_with_mods,text_features = model.compute_loss_with_extra_data(img1,
                                                        mods, 
                                                        img2, 
                                                        extra_data, 
                                                        soft_triplet_loss=True)
        
      # mitstates case
      elif opt.loss == 'batch_based_classification':
        if opt.model == 'tirg_evolved':
            loss_value = model.compute_loss_with_extra_data(img1, 
                                                            mods, 
                                                            img2, 
                                                            extra_data, 
                                                            soft_triplet_loss=False) # original approach
        else:
            loss_value = model.compute_loss(img1, 
                                            mods, 
                                            img2, 
                                            soft_triplet_loss=False)
      else:
        print 'Invalid loss function', opt.loss
        sys.exit()
        
      positive = cosine_loss(repr_to_compare_with_source, img1, fake_minus)
      positive_text = cosine_losstxt(repr_to_compare_with_mods, text_features, valid)
      # push away from source
      # negative = cosine_loss(repr_to_compare_with_source, img1, fake_minus)
        
      cos_loss = positive
      cos_loss_text = positive_text
      loss_name = opt.loss
      loss_weight = 1.0
      losses += [(loss_name, loss_weight, loss_value.cuda())]
      losses += [("cos_loss", 0.1, cos_loss)]
      losses += [("cos_loss_text", 0.1, cos_loss_text)]
      total_loss = sum([
          loss_weight * loss_value
          for loss_name, loss_weight, loss_value in losses
      ])
      assert not torch.isnan(total_loss)
      losses += [('total training loss', None, total_loss.item())]

      # track losses
      for loss_name, loss_weight, loss_value in losses:
        if not losses_tracking.has_key(loss_name):
          losses_tracking[loss_name] = []
        losses_tracking[loss_name].append(float(loss_value))

      # gradient descend
      optimizer.zero_grad()
      total_loss.backward()
      optimizer.step()

    for data in tqdm(trainloader, desc='Training for epoch ' + str(epoch)):
      it += 1
      training_1_iter(data)

      # decay learing rate
      if it >= opt.learning_rate_decay_frequency and it % opt.learning_rate_decay_frequency == 0:
        for g in optimizer.param_groups:
          g['lr'] *= 0.1

  print 'Finished training'


def main():
  opt = parse_opt()
  print 'Arguments:'
  for k in opt.__dict__.keys():
    print '    ', k, ':', str(opt.__dict__[k])


  import socket
  import os
  from datetime import datetime
  current_time = datetime.now().strftime('%b%d_%H-%M-%S')
  logdir = os.path.join(opt.log_dir, current_time + '_' + socket.gethostname() + opt.comment)

  logger = SummaryWriter(logdir)
  print 'Log files saved to', logger.file_writer.get_logdir()
  for k in opt.__dict__.keys():
    logger.add_text(k, str(opt.__dict__[k]))

  trainset, testset = load_dataset(opt)
  model, optimizer, scheduler = create_model_and_optimizer(
      opt, [t.decode('utf-8') for t in trainset.get_all_texts()])
        
  train_loop(opt, logger, trainset, testset, model, optimizer, scheduler)
  logger.close()


if __name__ == '__main__':
  main()
