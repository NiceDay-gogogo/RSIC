#!/usr/bin/env python
# coding: utf-8

# In[1]:


import json

# coco2014val文件
cocoJsonFile = './coco-caption/annotations/captions_val2014.json'
rsicdJsonFile = './data/Sydney/dataset_sydney.json' 


# In[2]:


coco_dict = json.load(open(cocoJsonFile, 'rb'))
rsicd_dict_old = json.load(open(rsicdJsonFile,'rb'))
rsicd_dict = {}


# In[3]:


for key, val in coco_dict.items():
    print(key)
    print("type of the key's val: ", type(val))
    print("len of the key's val ",len(val), '\n')


# In[4]:


for key, val in rsicd_dict_old.items():
    print(key)
    print("type of the key's val: ", type(val))
    print("len of the key's val ",len(val), '\n')


# In[6]:


rsicd_dict_old['dataset']


# In[5]:


# 处理 key 'info'
info = coco_dict['info']
# print(info)
info['description'] = 'This is stable 1.0 version of the RSICD dataset.'
info['url'] = 'http://GFZShiwai.org'
info['year'] = '2021'
info['contributor']  = 'XuLiangyu'
info['date_created'] = '2021-04-03 13:52:01'

# 生成RSICD dataset info 信息
rsicd_dict['info'] = info
rsicd_dict['info']


# In[7]:


# 处理 key 'images'
images_rsicd = []
# dict_keys(['filename', 'imgid', 'sentences', 'split', 'sentids'])   = images_rsicd_raw[0]
images_rsicd_raw = rsicd_dict_old['images']
images_coco = coco_dict['images']
# print(images_coco[0])
# {'license': 3, 'url': 'http://farm9.staticflickr.com/8186/8119368305_4e622c8349_z.jpg', 'file_name': 'COCO_val2014_000000391895.jpg', 'id': 391895, 'width': 640, 'date_captured': '2013-11-14 11:18:45', 'height': 360}
count = 0
for img in images_rsicd_raw:
    if img['split'] == 'val':
        count += 1
        img_dict = {}
        img_dict['license'] = 6
        img_dict['url'] = 'http://GFZShiwai.org'
        img_dict['file_name'] = img['filename']
        img_dict['id'] = img['imgid']
        img_dict['width'] = 124
        img_dict['date_captured'] = '2021-04-03 14:20:38'
        img_dict['height'] = 124
        images_rsicd.append(img_dict)
rsicd_dict['images'] = images_rsicd
print("The key 'images' done!", count)





# In[8]:


# 处理 key 'licenses'
# print(coco_dict['licenses'])
rsicd_dict['licenses'] = coco_dict['licenses']
print(rsicd_dict['licenses'])


# In[9]:


# 处理 key 'type'
print(coco_dict['type'])
rsicd_dict['type'] = 'captions'
print(rsicd_dict['type'])


# In[10]:


# 处理 key 'annotations'
rsicd_anno = []
print(coco_dict['annotations'][0])
count = 0
for img in images_rsicd_raw:
    if img['split'] == 'val':
        #print(img)
        for sentence in img['sentences']:
            count += 1
            #print('ok')
            anno_dict = {}
            anno_dict['image_id'] = sentence['imgid']
            anno_dict['id'] = sentence['sentid']
            anno_dict['caption'] = sentence['raw']
            rsicd_anno.append(anno_dict)
rsicd_dict['annotations'] = rsicd_anno
print("The key annotations done!", count)



# In[12]:


print(len(rsicd_dict['annotations']))


# In[13]:


newJsonFile = './coco-caption/annotations/captions_sydney_val.json'
json.dump(rsicd_dict, open(newJsonFile, 'w'))
