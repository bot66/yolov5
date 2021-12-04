import json
from collections import OrderedDict
import os
import yaml
import shutil
from tqdm import tqdm

# convert MSCOCO json to yolov5 label format

def create_dirs(dirs):
    for d in dirs:
        if not os.path.exists(d):
            os.makedirs(d)


def load_json(json_path):
    with open(json_path, 'r') as f:
        json_data = json.load(f)

    return json_data 

def normalize_bbox(img_dim,bbox):
    """
    img_dim: [w,h]

    bbox: [xmin, ymin, w, h]

    return: normalized center_x, center_y, w, h
    """

    xmin, ymin, w, h = bbox

    center_x = xmin + w/2
    center_y = ymin + h/2

    # normalize
    center_x = center_x/img_dim[0]
    center_y = center_y/img_dim[1]
    w = w/img_dim[0]
    h = h/img_dim[1]

    norm_bbox = [center_x, center_y, w, h]

    return norm_bbox

def write_result(dirs,res, tag = None):
    assert tag, "tag not being anounced !"
    
    if tag == "train":
        out_dir = dirs[2]
    elif tag == "val":
        out_dir = dirs[3]
    
    for file,bboxes in res.items():
        filename = file.split('.')[0] + '.txt'
        with open( os.path.join(out_dir, filename), 'w') as f:
            for bbox in bboxes:
                f.write(bbox + '\n')

def write_yaml(json_data):
    cato = json_data["categories"]
    cato = sorted(cato, key = lambda x : x['id'])
    nc = len(cato)
    names = []

    for c in cato:
        names.append(c["name"])

    train = os.path.abspath("dataset/images/train")
    val = os.path.abspath("dataset/images/val")

    yaml_data = {}
    yaml_data["train"]  = train
    yaml_data["val"]  = val
    yaml_data["nc"]  = nc
    yaml_data["names"]  = names

    with open('dataset/data.yaml', 'w') as f:
        yaml.dump(yaml_data, f)


def main():
    dirs = ["dataset/images/train", "dataset/images/val", "dataset/labels/train", "dataset/labels/val"]

    create_dirs(dirs)

    tags = ["train", "val"]

    for tag in tags:

        json_path = "coco-person/annotations_trainval2017/instances_{}2017.json".format(tag)
        
        if not os.path.exists(json_path):
            print("no " + tag + ".json")
            continue

        json_data = load_json(json_path)

        anns = sorted(json_data["annotations"], key=lambda x: x["image_id"])
        imgs = sorted(json_data["images"], key=lambda x: x["id"])
        imgs = dict(zip([x["id"] for x in imgs],imgs))

        print("processing {} data...".format(tag))
        res = {}
        for i, ann in tqdm(enumerate(anns)):
            bbox = ann["bbox"]
            image_id = ann["image_id"]
            img_info = imgs[image_id]

            img_dim = img_info["width"], img_info["height"]
            filename = img_info["file_name"]
            #copy image
            source_root = "coco-person/{}2017".format(tag)
            image_source = os.path.join(source_root,filename)
            dst_root = "dataset/images/{}/".format(tag)
            if not os.path.exists(os.path.join(dst_root,filename)):
                shutil.copy(image_source,dst_root)

            _ = res.setdefault(filename, [])

            class_id = ann["category_id"] - 1
            norm_bbox = normalize_bbox(img_dim,bbox)
            
            norm_bbox.insert(0,class_id)

            line = ' '.join([str(x) for x in norm_bbox])

            res[filename].append(line)

        write_result(dirs,res, tag = tag)
        
        write_yaml(json_data)


if __name__ == "__main__":
    main()


