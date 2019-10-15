# coding: utf-8

import xml.etree.ElementTree as ET
import os


def parse_xml(path):
    tree = ET.parse(path)
    img_name = path.split('/')[-1][:-4]

    height = tree.findtext("./size/height")
    width = tree.findtext("./size/width")

    objects = [img_name, width, height]

    for obj in tree.findall('object'):
        difficult = obj.find('difficult').text
        if difficult == '1':
            continue
        name = obj.find('name').text
        bbox = obj.find('bndbox')
        xmin = bbox.find('xmin').text
        ymin = bbox.find('ymin').text
        xmax = bbox.find('xmax').text
        ymax = bbox.find('ymax').text

        name = str(names_dict[name])
        objects.extend([name, xmin, ymin, xmax, ymax])
    if len(objects) > 1:
        return objects
    else:
        return None


names_dict = {}
cnt = 0
f = open('./voc_names.txt', 'r').readlines()
for line in f:
    line = line.strip()
    names_dict[line] = cnt
    cnt += 1

path = '/home/shihaobing/copy_from_disk/yolo/voc/'# xml文档的位置
file = os.listdir(path)
global train_cnt
train_cnt = 0
f = open('train.txt', 'w')
for img_name in file:
    img_name = img_name.strip()
    xml_path = path + img_name
    objects = parse_xml(xml_path)
    if objects:
        objects[0] = '/home/shihaobing/copy_from_disk/RSODDataset/'+ img_name.split('.')[0] + '.jpg'
        if os.path.exists(objects[0]):
            objects.insert(0, str(train_cnt))
            train_cnt += 1
            objects = ' '.join(objects) + '\n'
            f.write(objects)
f.close()
