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
from bert_serving.client import BertClient

bc = BertClient()


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

    # compute test query features
    imgs = []
    mods = []
    extra_data = []
    count = -1
    for t in tqdm(test_queries):
      count +=1
      if count == 5024: #157 batches
        break
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
            f, _, _, _, _, _ = model.compose_img_text_with_extra_data(imgs.cuda(), mods, extra_data, [])
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

  else:
    # use training queries to approximate training retrieval performance
    imgs0 = []
    imgs = []
    mods = []
    extra_data = []
    for i in range(96):
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
            f, _, _, _, _, _ = model.compose_img_text_with_extra_data(imgs.cuda(), mods, extra_data, [])
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

  all_imgs_needed = all_imgs
  # feature normalization
  for i in range(all_queries.shape[0]):
    all_queries[i, :] /= np.linalg.norm(all_queries[i, :])
  for i in range(all_imgs.shape[0]):
    all_imgs[i, :] /= np.linalg.norm(all_imgs[i, :])

  # match test queries to target images, get nearest neighbors
  sims = all_queries.dot(all_imgs.T)
  # if test_queries:
  #   for i, t in enumerate(test_queries):
  #     sims[i, t['source_img_id']] = -10e10  # remove query image
  nn_result = [np.argsort(-sims[i, :])[:110] for i in range(sims.shape[0])]
  COMB_recall_top = False

  print "Now doing stuff"
  del all_imgs
  if test_queries:
    all_queries_source_alltest = []
    imgs = []
    mods = []
    extra_data = []
    ding = torch.from_numpy(all_imgs_needed).cuda()
    mapped_image_features_target = model.image_features_mapping(ding)
    del all_imgs_needed
    all_imgs_needed =[]
    sims_sym_loss=[]
    for i, t in enumerate(test_queries):
      if i == 5024:
        break
      torch.cuda.empty_cache()
      # Here we calculate the q(target_image,source_caption)
      # Here target_images is complete set of test images
      # And source_caption is batch_size ... associated with batch_size set of test_source_images query
      # size of f is (batchsize, num_features) ... i.e. (32,512)
      # size of q is (1, num_features, len_test_images) ... i.e. (32,512,len(testset.imgs))
      # source_adjs = str(t["target_caption"]).split(' ')[0]  # [x.split(' ')[0] for x in str(t["source_caption"])]
      # source_captions = [adj + " " + noun for adj, noun in zip(source_adjs, str(t["noun"]))]
      source_captions = [t['mod']['str'].decode('utf-8')] #[source_adjs + " " + str(t["noun"])]
      source_captions = bc.encode(source_captions)
      source_captions = torch.from_numpy(source_captions).cuda()

      mapped_text_features_source = model.text_features_mapping(source_captions)
      repeated_text_features = mapped_text_features_source.data[0].unsqueeze(0).expand(10546,512)
      f1_image2 = model.image_gated_feature_composer((mapped_image_features_target,
                                                      repeated_text_features))
      f2_image2 = model.image_res_info_composer((mapped_image_features_target,
                                                 repeated_text_features))
      f_image2 = torch.sigmoid(f1_image2) * ding * model.c[0] + model.c[1] * f2_image2
      f_image2 = f_image2.data.cpu().numpy()
      all_queries_source_alltest += [f_image2]
      repeated_text_features=[]
      f_image2=[]
      del mapped_text_features_source

      all_queries_source_alltest = np.concatenate(all_queries_source_alltest)
      # feature normalization
      for j in range(all_queries_source_alltest.shape[0]):
        all_queries_source_alltest[j, :] /= np.linalg.norm(all_queries_source_alltest[j, :])#????
        # the operation here is diff than above
        # here you have 10546,512 shape thing?
      sims_brushed_bush_to_mossy_bush = all_queries[i,:].dot(all_queries_source_alltest.T)

      # sims_brushed_bush_to_mossy_bush[0, t['source_img_id']] = -10e10  # remove query image
      all_queries_source_alltest=[]
      sims_sym_loss += [sims_brushed_bush_to_mossy_bush]
      sims_brushed_bush_to_mossy_bush = []

    try:
      print len(sims_sym_loss)
      print sims_sym_loss[-1].shape
    except:
      print "its fine"
    sims_sym_loss = np.concatenate(sims_sym_loss)
    # sims_sym_loss is also, I think ... 82724 x 10546 here
    print "Saving the numpy files"
    np.save('sims_sym_loss_5024', sims_sym_loss)
    print "Saved sims_sym_loss files"
    np.save('sims_5024', sims)
    print "Saved sims files"
    np.save('all_captions_5024', all_captions)
    print "Saved all_captions files"
    np.save('all_target_captions_5024', all_target_captions)
    print "Saved all_target_captions files"

    sims_sym_loss = np.reshape(sims_sym_loss, sims.shape)
    COMB_recall_top = False
    try:
      print (sims_sym_loss.shape)
      print (sims.shape)
      assert (sims_sym_loss.shape == sims.shape)
      sims_comb = sims + sims_sym_loss
      nn_result_comb = [np.argsort(-sims_comb[i, :]) for i in range(sims_comb.shape[0])]
      nn_result_comb = [[all_captions[nn] for nn in nns] for nns in nn_result_comb]
      COMB_recall_top = True
    except:
      print "This sims_comb = sims + sims_sym_loss can not be done "
      COMB_recall_top = False
    nn_result_sym = [np.argsort(-sims_sym_loss[i, :]) for i in range(sims_sym_loss.shape[0])]
    nn_result_sym = [[all_captions[nn] for nn in nns] for nns in nn_result_sym]


  # sims is 82724 x 10546 here
  # sims = all_queries.dot(all_imgs.T)
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
    r = 0.0
    if test_queries:
      for i, nns in enumerate(nn_result_sym):
        if all_target_captions[i] in nns[:k]:
          r += 1
      r /= len(nn_result)
      out += [('SYM_recall_top' + str(k) + '_correct_composition', r)]
      if COMB_recall_top:
        r = 0.0
        for i, nns in enumerate(nn_result_comb):
          if all_target_captions[i] in nns[:k]:
            r += 1
        r /= len(nn_result)
        out += [('COMB_recall_top' + str(k) + '_correct_composition', r)]


  return out
# python main.py --dataset=mitstates --dataset_path=../data/release_dataset/ --model=tirg_evolved --loss=soft_triplet --learning_rate_decay_frequency=50000 --num_iters=160000 --weight_decay=5e-5 --comment=mitstates_tirg_evolved_symloss --log_dir ../logs/mitstates/ --model_checkpoint ../logs/mitstates/Mar23_20-45-38_ip-172-31-38-215mitstates_tirg_evolved_bert_no_resids_text_loss_joined_norm_scaled_1and4_ab/latest_checkpoint.pth --use_pretrained True --test_only True
