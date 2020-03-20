# Copyright 2018 Google Inc. All Rights Reserved.
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

"""Evaluates the retrieval model."""
import numpy as np
import ast
import torch
import random
from tqdm import tqdm as tqdm


def test(opt, model, testset):
  """Tests a model over the given testset."""
  model.eval()
  test_queries = testset.get_test_queries()
  print('test_queries', len(test_queries))
   
  all_imgs = []
  all_captions = []
  all_queries = []
  all_target_captions = []
  if test_queries:
    if opt.dataset == "mitstates_regions":
        print('sampling 20000')
        test_queries = random.sample(test_queries, 20000)
    # compute test query features
    imgs = []
    mods = []
    extra_data = []
    for t in tqdm(test_queries):
      torch.cuda.empty_cache()
      imgs += [testset.get_img(t['source_img_id'])]
      mods += [t['mod']['str']]
      if (opt.dataset == 'fashion200k'):
            if not extra_data:
                extra_data = [[]]
            extra_data[0].append(str(t['target_caption']))
      elif (opt.dataset == 'css3d'):
        # extra_data += [t['mod']['str']]
        objects = t['source_img_objects']
        obj_data = []
        for obj in objects:
            if obj['shape']:
                obj_data.append(obj['pos_str'] + ' ' + obj['size'] + ' ' + obj['color'] + ' ' + obj['shape'])
                
        target_objects = t['target_img_objects']
        target_obj_data = []
        for obj in target_objects:
            if obj['shape']:
                target_obj_data.append(obj['pos_str'] + ' ' + obj['size'] + ' ' + obj['color'] + ' ' + obj['shape'])
                
        if not extra_data:
            extra_data = [[], []]
        extra_data[0].append(target_obj_data)
        extra_data[1].append(obj_data)
      else:
        if not extra_data:
            extra_data = [[] for i in range(4)]
        extra_data[0].append(str(t["noun"]))
        extra_data[1].append(ast.literal_eval(t["context_classes"]))
        extra_data[2].append(ast.literal_eval(t["context_classes"]))
        extra_data[3].append(str(t["source_caption"]))
        
      if len(imgs) >= opt.batch_size or t is test_queries[-1]:
        if 'torch' not in str(type(imgs[0])):
          imgs = [torch.from_numpy(d).float() for d in imgs]
        imgs = torch.stack(imgs).float()
        imgs = torch.autograd.Variable(imgs).cuda()
        mods = [t.decode('utf-8') for t in mods]
        if opt.model in ['tirg_evolved', 'tirg_lastconv_evolved']:
            f, _, _ = model.compose_img_text_with_extra_data(imgs.cuda(), mods, extra_data)
            # f = model.compose_img_text_with_extra_data(imgs.cuda(), mods, extra_data)
            f = f.data.cpu().numpy()
        else:
            f = model.compose_img_text(imgs.cuda(), mods).data.cpu().numpy()
        all_queries += [f]
        imgs = []
        mods = []
        extra_data = []
    all_queries = np.concatenate(all_queries)
    all_target_captions = [t['target_caption'] for t in test_queries]

    # compute all image features
    imgs = []
    for i in tqdm(range(len(testset.imgs))):
      imgs += [testset.get_img(i)]
      if len(imgs) >= opt.batch_size or i == len(testset.imgs) - 1:
        if 'torch' not in str(type(imgs[0])):
          imgs = [torch.from_numpy(d).float() for d in imgs]
        imgs = torch.stack(imgs).float()
        imgs = torch.autograd.Variable(imgs).cuda()
        imgs = model.extract_img_feature(imgs.cuda()).data.cpu().numpy()
        
        all_imgs += [imgs]
        imgs = []
    all_imgs = np.concatenate(all_imgs)
    all_captions = [img['captions'][0] for img in testset.imgs]

  else:
    # use training queries to approximate training retrieval performance
    imgs0 = []
    imgs = []
    mods = []
    extra_data = []
    for i in range(10000):
      torch.cuda.empty_cache()
      item = testset[i]
      imgs += [item['source_img_data']]
      if (opt.dataset == 'fashion200k'):
            if not extra_data:
                extra_data = [[]]
            extra_data[0].append(str(item['target_caption']))
      elif (opt.dataset == 'css3d'):
        # extra_data += [t['mod']['str']]
        objects = item['source_img_objects']
        obj_data = []
        for obj in objects:
            if obj['shape']:
                obj_data.append(obj['pos_str'] + ' ' + obj['size'] + ' ' + obj['color'] + ' ' + obj['shape'])
                
        target_objects = t['target_img_objects']
        target_obj_data = []
        for obj in target_objects:
            if obj['shape']:
                target_obj_data.append(obj['pos_str'] + ' ' + obj['size'] + ' ' + obj['color'] + ' ' + obj['shape'])
                
        if not extra_data:
            extra_data = [[], []]
        extra_data[0].append(target_obj_data)
        extra_data[1].append(obj_data)
      else:
        if not extra_data:
            extra_data = [[] for i in range(4)]
        extra_data[0].append(str(item["noun"]))
        extra_data[1].append(ast.literal_eval(item["context_classes"]))
        extra_data[2].append(ast.literal_eval(item["target_context_classes"]))
        extra_data[3].append(str(item["source_caption"]))
        
      mods += [item['mod']['str']]
      if len(imgs) > opt.batch_size or i == 9999:
        if type(imgs[0]).__module__ == 'numpy.core.memmap':
            imgs = torch.from_numpy(np.array(imgs)).cuda()
        else:
            imgs = torch.stack(imgs).float() ## !!!
        imgs = torch.autograd.Variable(imgs)
        mods = [t.decode('utf-8') for t in mods]
        # nouns = [t.decode('utf-8') for t in nouns]
        if opt.model in ['tirg_evolved', 'tirg_lastconv_evolved']:
            f, _, _ = model.compose_img_text_with_extra_data(imgs.cuda(), mods, extra_data)
            # f = model.compose_img_text_with_extra_data(imgs.cuda(), mods, extra_data)
            f = f.data.cpu().numpy()
        else:
            f = model.compose_img_text(imgs.cuda(), mods).data.cpu().numpy()
        all_queries += [f]
        imgs = []
        extra_data = []
        mods = []
      imgs0 += [item['target_img_data']]
      if len(imgs0) > opt.batch_size or i == 9999:
        if type(imgs0[0]).__module__ == 'numpy.core.memmap':
            imgs0 = torch.from_numpy(np.array(imgs0)).cuda()
        else:
            imgs0 = torch.stack(imgs0).float()
        imgs0 = torch.autograd.Variable(imgs0)
        imgs0 = model.extract_img_feature(imgs0.cuda()).data.cpu().numpy()
        all_imgs += [imgs0]
        imgs0 = []
      all_captions += [item['target_caption']]
      all_target_captions += [item['target_caption']]
    all_imgs = np.concatenate(all_imgs)
    all_queries = np.concatenate(all_queries)

  # feature normalization
  for i in range(all_queries.shape[0]):
    all_queries[i, :] /= np.linalg.norm(all_queries[i, :])
  for i in range(all_imgs.shape[0]):
    all_imgs[i, :] /= np.linalg.norm(all_imgs[i, :])

  # match test queries to target images, get nearest neighbors
  sims = all_queries.dot(all_imgs.T)
  if test_queries:
    for i, t in enumerate(test_queries):
      sims[i, t['source_img_id']] = -10e10  # remove query image
  nn_result = [np.argsort(-sims[i, :])[:110] for i in range(sims.shape[0])]

  # compute recalls
  out = []
  nn_result = [[all_captions[nn] for nn in nns] for nns in nn_result]
  for k in [1, 5, 10, 50, 100]:
    r = 0.0
    for i, nns in enumerate(nn_result):
      if all_target_captions[i] in nns[:k]:
        r += 1
    r /= len(nn_result)
    out += [('recall_top' + str(k) + '_correct_composition', r)]

    if opt.dataset == 'mitstates':
      r = 0.0
      for i, nns in enumerate(nn_result):
        if all_target_captions[i].split()[0] in [c.split()[0] for c in nns[:k]]:
          r += 1
      r /= len(nn_result)
      out += [('recall_top' + str(k) + '_correct_adj', r)]

      r = 0.0
      for i, nns in enumerate(nn_result):
        if all_target_captions[i].split()[1] in [c.split()[1] for c in nns[:k]]:
          r += 1
      r /= len(nn_result)
      out += [('recall_top' + str(k) + '_correct_noun', r)]

  return out
