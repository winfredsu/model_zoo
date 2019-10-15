import os
import shutil

TRAIN_SPLIT = 0.8

if os.path.isdir('Annotation'):
    shutil.rmtree('Annotation')

os.makedirs('Annotation/train')
os.makedirs('Annotation/val')

# all object names
# obj_names = os.listdir('dataset-original')
obj_names = ['oiltank']

for obj_name in obj_names:
    pathname = 'dataset-original/'+obj_name+'/Annotation/xml'
    xml_files = os.listdir(pathname)
    train_split = int(TRAIN_SPLIT*len(xml_files))
    train_xml_files = xml_files[:train_split]
    val_xml_files = xml_files[train_split:]

    for file in train_xml_files:
        shutil.copyfile(os.path.join(pathname,file), os.path.join('Annotation/train',file))

    for file in val_xml_files:
        shutil.copyfile(os.path.join(pathname,file), os.path.join('Annotation/val',file))

