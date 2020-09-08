import os, json
import numpy as np


def read_data(type_list=['train', 'validation']):
    root_path = '/home/jayeon'

    img_list = {}
    item_dict = {}

    base_path = os.path.join(root_path, 'DeepFashion2')
    for file_type in type_list:
        anno_dir_path = os.path.join(base_path, file_type, 'annos')
        img_list[file_type] = []
        item_dict[file_type] = {}
        item_idx = 0
        for file_name in os.listdir(os.path.join(base_path, file_type, 'image')):
            anno_path = os.path.join(anno_dir_path, file_name.split('.')[0] + '.json')
            if not os.path.exists(anno_path):
                continue
            anno = json.load(open(anno_path, 'r'))
            source_type = 0 if anno['source'] == 'user' else 1
            pair_id = str(anno['pair_id'])
            for key in anno.keys():
                if key not in ['source', 'pair_id'] and int(anno[key]['style']) > 0:
                    bounding_box = np.asarray(anno[key]['bounding_box'], dtype=int)
                    cate_id = anno[key]['category_id'] - 1
                    pair_style = '_'.join([pair_id, str(anno[key]['style'])])
                    if pair_style not in item_dict[file_type].keys():
                        item_dict[file_type][pair_style] = item_idx
                        item_idx += 1
                    img_list[file_type].append([os.path.join(file_type, 'image', file_name),
                                                item_dict[file_type][pair_style], cate_id,
                                                bounding_box, source_type])

        img_list[file_type] = np.asarray(img_list[file_type], dtype=object)

    return img_list, base_path, item_dict
