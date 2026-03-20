#!/usr/bin/env python
# coding: utf-8

# In[1]:


import json

# 输入文件
cocoJsonFile = './coco-caption/annotations/captions_val2014.json'
rsicdJsonFile = './data/Sydney/dataset_sydney.json' 

# 输出文件
newJsonFile = './coco-caption/annotations/captions_sydney_test.json'


# In[2]:


coco_dict = json.load(open(cocoJsonFile, 'rb'))
rsicd_dict_old = json.load(open(rsicdJsonFile,'rb'))
rsicd_dict = {}


# In[3]:


# 处理 key 'info'
info = coco_dict['info']
info['description'] = 'This is stable 1.0 version of the RSICD test dataset.'
info['url'] = 'http://GFZShiwai.org'
info['year'] = '2021'
info['contributor']  = 'XuLiangyu'
info['date_created'] = '2021-04-03 13:52:01'

# 生成RSICD dataset info 信息
rsicd_dict['info'] = info


# In[4]:


# 处理 key 'images' - 只选择test分割的数据
images_rsicd = []
images_rsicd_raw = rsicd_dict_old['images']
count = 0
for img in images_rsicd_raw:
    if img['split'] == 'test':
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


# In[5]:


# 处理 key 'licenses'
rsicd_dict['licenses'] = coco_dict['licenses']


# In[6]:


# 处理 key 'type'
rsicd_dict['type'] = 'captions'


# In[7]:


# 处理 key 'annotations' - 只处理test分割的句子
rsicd_anno = []
count = 0
for img in images_rsicd_raw:
    if img['split'] == 'test':
        for sentence in img['sentences']:
            count += 1
            anno_dict = {}
            anno_dict['image_id'] = sentence['imgid']
            anno_dict['id'] = sentence['sentid']
            anno_dict['caption'] = sentence['raw']
            rsicd_anno.append(anno_dict)
            
rsicd_dict['annotations'] = rsicd_anno
print("The key annotations done!", count)


# In[8]:


# 保存为测试集标注文件
json.dump(rsicd_dict, open(newJsonFile, 'w'))
print("Test annotations saved to", newJsonFile)


# In[ ]:



