from pprint import pprint
import os
import json
import os.path as osp
import sys
import sng_parser

# load data
def load_relation_dict():
    obj_head = {}
    rel_head = {}
    selected_obj_mapping = {}
    selected_relation_mapping = {}

    for line in open(os.path.join(DATA, "scanrefer_object_selected_head.txt"),"r"):
        name = ""
        for st in line.split(' ')[:-1]:
            name = (name + ' ' + st)
        obj_head[name.strip(' ')] = int(line.split(' ')[-1].strip('\n'))

    for line in open(os.path.join(DATA, "scanrefer_relation_selected_head.txt"),"r"):
        name = ""
        for st in line.split(' ')[:-1]:
            name = (name + ' ' + st)
        rel_head[name.strip(' ')] = int(line.split(' ')[-1].strip('\n'))

    for line in open(os.path.join(DATA, "scanrefer_object_selected.txt"),"r"):
        name = line.strip('\n').strip(' ')
        selected_obj_mapping[name] = "None"
        for n in obj_head.keys():
            if n in name:
                selected_obj_mapping[name] = n
                break
    selected_obj_mapping["trashcan"] = "trash can"

    for line in open(os.path.join(DATA, "scanrefer_relation_selected.txt"),"r"):
        name = line.strip('\n').strip(' ')
        selected_relation_mapping[name] = "None"
        for n in rel_head.keys():
            if n in name:
                selected_relation_mapping[name] = rel_head[n]
                break

    return obj_head, rel_head, selected_obj_mapping, selected_relation_mapping

def load_attribute_dict():
    attr_color = {}
    attr_shape = {}
    attr_size = {}

    for line in open(os.path.join(DATA, "attr_dict_color.txt"),"r"):
        color, id = line.splitlines()[0].split(' ')
        attr_color[color] = id
    print('color:', len(attr_color))
    for line in open(os.path.join(DATA, "attr_dict_shape.txt"),"r"):
        shape, id = line.splitlines()[0].split(' ')
        attr_shape[shape] = id
    print('shape:', len(attr_shape))
    for line in open(os.path.join(DATA, "attr_dict_size.txt"),"r"):
        size, id = line.splitlines()[0].split(' ')
        attr_size[size] = id
    print('size:', len(attr_size))

    return attr_color, attr_shape, attr_size

def process_relation(entities, relations, object_name):
    def is_in_object_list(object):
        return object.strip('\n').strip(' ') in selected_obj_mapping.keys()
    
    def is_in_relation_list(relation):
        return relation.strip('\n').strip(' ') in selected_relation_mapping.keys()

    relation_processed = []
    fail_reason_here = {}
    fail_object_here = []
    fail_relation_here = []
    for i, rel in enumerate(relations):
        rel_processed = {}
        if is_in_relation_list(rel["relation"]):
            if rel["subject"] is not None:
                sub_lemma_span = entities[rel["subject"]]["lemma_span"]
                sub_lemma_head = entities[rel["subject"]]["lemma_head"]
            else:
                sub_lemma_span = sub_lemma_head = object_name

            if rel["object"] is not None:
                obj_lemma_span = entities[rel["object"]]["lemma_span"]
                obj_lemma_head = entities[rel["object"]]["lemma_head"]
            else:
                obj_lemma_span = obj_lemma_head = object_name


            if (is_in_object_list(sub_lemma_head)) and (is_in_object_list(obj_lemma_head)):
                rel_processed["subject"] = {"sub_lemma_span": sub_lemma_span, "sub_lemma_head": selected_obj_mapping[sub_lemma_head]}
                rel_processed["object"] = {"obj_lemma_span": obj_lemma_span, "obj_lemma_head": selected_obj_mapping[obj_lemma_head]}
                rel_processed["relation"] = rel["relation"]
                rel_processed["relation_id"] = selected_relation_mapping[rel["relation"]]
        else:
            if rel["relation"] not in fail_reason_here.keys():
                fail_reason_here[rel["relation"]] = ["relation not in list"]
            else:
                fail_reason_here[rel["relation"]].append("relation not in list")
        if rel_processed:
            relation_processed.append(rel_processed)
    return relation_processed, fail_reason_here, fail_object_here, fail_relation_here

def process_ent_attrbution(ent):
    attr_list = {'color': [], 'shape': [], 'size': []}
    attr_list_id = {'color': [], 'shape': [], 'size': []}
    if 'modifiers' in ent:
        for i, modifier in enumerate(ent['modifiers']):
            if modifier['lemma_span'] in attr_color.keys():
                attr_list['color'].append(modifier['lemma_span'])
                attr_list_id['color'].append(attr_color[modifier['lemma_span']])
            elif modifier['lemma_span'] in attr_shape.keys():
                attr_list['shape'].append(modifier['lemma_span'])
                attr_list_id['shape'].append(attr_shape[modifier['lemma_span']])
            elif modifier['lemma_span'] in attr_size.keys():
                attr_list['size'].append(modifier['lemma_span'])
                attr_list_id['size'].append(attr_size[modifier['lemma_span']])
    ent.update({'attrbution': attr_list})
    ent.update({'attrbution_id': attr_list_id})

    # if 'modifiers' in ent:
    #     for i, modifier in enumerate(ent['modifiers']):
    #         if modifier['lemma_span'] in attr_dict:
    #             attr_dict[modifier['lemma_span']] += 1
    #         else:
    #             attr_dict[modifier['lemma_span']] = 1
    return ent

def process_obj_attr(obj, ents):
    _attr = {'color': [], 'shape': [], 'size': []}
    _attr_id = {'color': [], 'shape': [], 'size': []}
    for ent in ents:
        if 'modifiers' in ent and (ent['lemma_head'].replace(' ', '') in obj.replace('_', ' ').replace(' ', '') or obj.replace('_', ' ').replace(' ', '') in ent['lemma_head'].replace(' ', '')):
            _attr['color'].extend(ent["attrbution"]['color'])
            _attr['shape'].extend(ent["attrbution"]['shape'])
            _attr['size'].extend(ent["attrbution"]['size'])
            _attr_id['color'].extend(ent["attrbution_id"]['color'])
            _attr_id['shape'].extend(ent["attrbution_id"]['shape'])
            _attr_id['size'].extend(ent["attrbution_id"]['size'])
    _attr['color'] = list(set(_attr['color']))
    _attr['shape'] = list(set(_attr['shape']))
    _attr['size'] = list(set(_attr['size']))
    _attr_id['color'] = list(set(_attr_id['color']))
    _attr_id['shape'] = list(set(_attr_id['shape']))
    _attr_id['size'] = list(set(_attr_id['size']))
    # if len(_attr['color']) > 1:
    #     _attr['color'] = []
    #     _attr_id['color'] = []
    # if len(_attr['shape']) > 1:
    #     _attr['shape'] = []
    #     _attr_id['shape'] = []
    # if len(_attr['size']) > 1:
    #     _attr['size'] = []
    #     _attr_id['size'] = []
    return _attr, _attr_id, (len(_attr['color'])+len(_attr['shape'])+len(_attr['size'])) != 0, len(_attr['color']) < 2 and len(_attr['shape']) < 2 and len(_attr['size']) < 2

def process_single_info(data):   
    object_name = data["object_name"]
    object_name = object_name.replace('_', ' ').replace('trash can', 'trashcan')

    des = data["description"]
    des_replace = des.replace(' it ', ' ' + 'the ' + object_name + ' ').replace('it ', 'the ' + object_name + ' ').replace(' it.', ' ' + 'the ' + object_name + '.').replace(' this is ', ' ' + 'the ' + object_name + ' ').replace('this is ', 'the ' + object_name + ' ').replace(' they are ', ' ' + 'the ' + object_name + ' ').replace('they are ', 'the ' + object_name + ' ').replace('this ', ' ').replace(' this ', ' ')
    des_replace = des_replace.replace('trash can', 'trashcan')
    des_replace = des_replace.replace('there is ', ' ').replace('there are ', ' ').replace(' is ', ' ')

    single_info = {"scene_id": data["scene_id"], "object_id": data["object_id"], "object_name": data["object_name"], "ann_id": data["ann_id"], "description": data["description"], "token": data["token"], "relation": []}

    assert des_replace
    graph, entities, relations = sng_parser.parse(des_replace)

    if not relations:
        rel_failed_info = {**single_info, 'fail_reason': "no relation"}
    else:
        relation_processed, fail_reason_here, fail_object_here, fail_relation_here = process_relation(entities, relations, object_name)
        if relation_processed:
            single_info = {**single_info, "relation": relation_processed}
            rel_failed_info = {}
        else:
            rel_failed_info = {**single_info, 'fail_reason': fail_reason_here, 'fail_object_here': fail_object_here, 'fail_relation_here': fail_relation_here}

    for ent in entities:
        process_ent_attrbution(ent)
    object_attr, object_attr_id, has_attr, is_unique = process_obj_attr(data["object_name"], entities)

    if not has_attr:
        attr_failed_info = {**single_info, 'fail_reason': '0 attrbute'}
    else:
        attr_failed_info = {}
    single_info = {**single_info, "object_attrbution":object_attr, "object_attrbution_id":object_attr_id, }

    return single_info, rel_failed_info, attr_failed_info

def process_data(file_name):
    file_path = os.path.join(DATA, file_name+'.json')
    data = json.load(open(file_path))

    info = []
    rel_failed_info = []
    
    rel_count = 0
    attr_count = 0
    for i, item in enumerate(data):
        single_info, single_rel_failed_info, single_attr_failed_info = process_single_info(item)
        info.append(single_info)
        if single_rel_failed_info == {}:
            rel_count += 1
        else:
            rel_failed_info.append(single_rel_failed_info)
        if single_attr_failed_info == {}:
            attr_count += 1

        # if i>100:
        #     break
        if i%100==0:
            print(i)

    with open(os.path.join(DATA, file_name+'_parser.json'), "w") as f:
        f.write(json.dumps(info, indent=4))
    with open(os.path.join(DATA, file_name+'_rel_failinfo.json'), "w") as f:
        f.write(json.dumps(rel_failed_info, indent=4))
    print("number of total cases: ", len(info))
    print("number of total relations: ", rel_count)
    print("number of total attrbutes: ", attr_count)

if __name__== "__main__":
    DATA = "./data/"
    SCANREFER_TRAIN = "ScanRefer_filtered_train"
    SCANREFER_VAL = "ScanRefer_filtered_val"

    obj_head, rel_head, selected_obj_mapping, selected_relation_mapping = load_relation_dict()
    attr_color, attr_shape, attr_size = load_attribute_dict()

    process_data(SCANREFER_TRAIN)
    process_data(SCANREFER_VAL)

    