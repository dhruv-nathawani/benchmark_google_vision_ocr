import glob
import io
import json
import os
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw
from google.cloud import vision
from joblib import Parallel, delayed
import xml.etree.ElementTree as ET
from tqdm import tqdm
from joblib import Parallel, delayed
from math import sqrt
from collections import defaultdict
from itertools import compress


def detect_text_google_vision(path):
    """Detects text in the file."""
    client = vision.ImageAnnotatorClient()
    with io.open(path, 'rb') as image_file:
        content = image_file.read()
    image = vision.types.Image(content=content)
    response = client.text_detection(image=image)
    return response.text_annotations


def PolyArea(x, y):
    return 0.5*np.abs(np.dot(x, np.roll(y, 1))-np.dot(y, np.roll(x, 1)))


def box_iou(box1, box2):
    '''Compute the intersection over union of two set of boxes.
    The box order must be (xmin, ymin, xmax, ymax).
    Args:
      box1: (tensor) bounding boxes, sized [N,4].
      box2: (tensor) bounding boxes, sized [M,4].
    Return:
      (tensor) iou, sized [N,M].
    Reference:
      https://github.com/kuangliu/torchcv/blob/master/torchcv/utils/box.py
    '''
    N = box1.size(0)
    M = box2.size(0)

    lt = torch.max(box1[:, None, :2], box2[:, :2])  # [N,M,2]
    rb = torch.min(box1[:, None, 2:], box2[:, 2:])  # [N,M,2]

    wh = (rb-lt).clamp(min=0)      # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    area1 = (box1[:, 2]-box1[:, 0]) * (box1[:, 3]-box1[:, 1])  # [N,]
    area2 = (box2[:, 2]-box2[:, 0]) * (box2[:, 3]-box2[:, 1])  # [M,]
    iou = inter / (area1[:, None] + area2 - inter)
    return iou


def plot_google_result(image_path, google_response):
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)
    for result in google_response[1:]:
        points = [(vertex.x, vertex.y)
                  for vertex in result.bounding_poly.vertices]
        draw.polygon(points, outline=(255, 0, 0))
    image.show()


def plot_gt_pred_bbox(image_path, pred_bbox, gt_bbox):
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)
    for result in pred_bbox:
        box = (result[0], result[1], result[2], result[3])
        draw.rectangle(box, outline=(0))
    image.show()
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)
    for result in gt_bbox:
        draw.rectangle(result, outline=(0))
    image.show()


def convert_bbox_to_xyxy(bbox):
    return [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]


def convert_bbox_list_to_xyxy(bbox_list):
    return [convert_bbox_to_xyxy(bbox) for bbox in bbox_list]


def get_rectangle_from_polygon(polygon):
    'Input format[x, y, x, y..]'
    x = polygon[::2]
    y = polygon[1::2]
    return [min(x), min(y), max(x), max(y)]


def check_bounding_box(inner_bbox, outer_bbox):
    # If top-left inner box corner is inside the bounding box
    if outer_bbox[0] <= inner_bbox[0] and outer_bbox[1] <= inner_bbox[1]:
        # If bottom-right inner box corner is inside the bounding box
        if inner_bbox[2] <= outer_bbox[2] and inner_bbox[3] <= outer_bbox[3]:
            return True
        else:
            return False


def get_google_bbox(google_response, hw_rect):
    google_bbox = []
    google_bbox_word_predictions = []
    for result in google_response[1:]:
        polygon = []
        for vertex in result.bounding_poly.vertices:
            polygon.append(vertex.x)
            polygon.append(vertex.y)
        rect = get_rectangle_from_polygon(polygon)
        if check_bounding_box(rect, hw_rect):
            google_bbox.append(get_rectangle_from_polygon(polygon))
            google_bbox_word_predictions.append(result.description)
    return google_bbox, google_bbox_word_predictions


def levenshtein(u, v):
    prev = None
    curr = [0] + list(range(1, len(v) + 1))
    # Operations: (SUB, DEL, INS)
    prev_ops = None
    curr_ops = [(0, 0, i) for i in range(len(v) + 1)]
    for x in range(1, len(u) + 1):
        prev, curr = curr, [x] + ([None] * len(v))
        prev_ops, curr_ops = curr_ops, [(0, x, 0)] + ([None] * len(v))
        for y in range(1, len(v) + 1):
            delcost = prev[y] + 1
            addcost = curr[y - 1] + 1
            subcost = prev[y - 1] + int(u[x - 1] != v[y - 1])
            curr[y] = min(subcost, delcost, addcost)
            if curr[y] == subcost:
                (n_s, n_d, n_i) = prev_ops[y - 1]
                curr_ops[y] = (n_s + int(u[x - 1] != v[y - 1]), n_d, n_i)
            elif curr[y] == delcost:
                (n_s, n_d, n_i) = prev_ops[y]
                curr_ops[y] = (n_s, n_d + 1, n_i)
            else:
                (n_s, n_d, n_i) = curr_ops[y - 1]
                curr_ops[y] = (n_s, n_d, n_i + 1)
    return curr[len(v)], curr_ops[len(v)]


def compute_score(image_path, gt_dict):
    image_name = os.path.split(image_path)[1]
    #xml_file = '-'.join(image_name.split('.')[0].split('-')[:-2])
    # xml_file = image_name.split('.')[0]
    # xml_file = os.path.join(gt_data_path, xml_file) + '.xml'
    # root = ET.parse(xml_file).getroot()
    #
    # gt_image_bbox = []
    #
    # min_x, min_y, max_x, max_y = 0, 0, 0, 0
    #
    # all_words = root.findall('handwritten-part/line/word')
    # for i, type_tag in enumerate(all_words):
    #     #if type_tag.get('id') == image_name.split('.')[0]:
    #     if type_tag.get('id').startswith(image_name.split('.')[0]):
    #         gt_text = type_tag.get('text')
    #         #gt_image_bbox = type_tag.findall('cmp')
    #         for t in type_tag.findall('cmp'):
    #             x1 = int(t.get('x'))
    #             y1 = int(t.get('y'))
    #             w = int(t.get('width'))
    #             h = int(t.get('height'))
    #             x2 = x1 + w
    #             y2 = y1 + h
    #             gt_image_bbox.append([x1, y1, x2, y2])
    #             if i == 0:
    #                 min_x, min_y = x1, y1
    #             if i == len(all_words) - 1:
    #                 max_x, max_y = x2, y2
    # hw_bbox = [min_x, min_y, max_x, max_y]


    #label_path = os.path.join(
    #    gt_data_path, 'gt_' + image_name[:-3] + 'txt')
    #with open(label_path) as f:
    #    content = f.readlines()
    #gt_split = []
    #for gt in content:
    #    gt_split.append(gt.replace(u'\ufeff', '').replace(' ', '').split(','))

    # remove blocks marked as "do not care"
    #if gt.count(',') > 6:
    #    size = 8
    #else:
    #    size = 4
    #gt_image_bbox = [get_rectangle_from_polygon(list(map(int, x[:size])))
    #                 for x in gt_split if x[-1].rstrip() != '###']

    google_response = detect_text_google_vision(image_path)
    if not google_response:
        return '', 0, 0 #len(gt_image_bbox)

    #print(google_response)

    #google_bbox = get_google_bbox(google_response)

    #print(gt_image_bbox)
    #print(google_bbox)

    #ious = box_iou(torch.from_numpy(np.array(gt_image_bbox).astype(
    #    'float')), torch.from_numpy(np.array(google_bbox).astype('float')))

    #tp = (ious.max(dim=1)[0] > 0.5).sum().item()
    #fp = (ious.max(dim=0)[0] < 0.5).sum().item()
    #fn = len(gt_image_bbox) - tp
    return google_response


def word_image_benchmarking():
    img_dir = '/Users/dnathawani/Desktop/hwt/words'
    gt_data_path = '/Users/dnathawani/Desktop/hwt/xml'

    test_split_file = '/Users/dnathawani/Desktop/hwt/largeWriterIndependentTextLineRecognitionTask/testset.txt'
    test_indices = []
    with open(test_split_file, 'r') as f:
        for line in f:
            line = line.strip('\n')
            test_indices.append(line)

    image_list = []
    #for ext in ['*.png', '*.jpg']:
    for root, dirs, files in os.walk(img_dir):
        if not dirs:
            for file in files:
                file_id = file.split('.')[0]
                file_id = file_id[:-3]
                if file_id in test_indices:
                    image_list.append(os.path.join(root, file))

    # with open(gt_json_path) as fd:
    #     gt_json = json.load(fd)
    # gt_annotations_df = pd.DataFrame(gt_json['annotations'])
    # gt_images_df = pd.DataFrame(gt_json['images'])
    # gt_df = gt_annotations_df.merge(
    #     gt_images_df, left_on='image_id', right_on='id')

    image_list = image_list[:50]

    tp_total, fp_total, fn_total = 0., 0., 0.
    cer_s = 0
    cer_i = 0
    cer_d = 0
    cer_n = 0
    n_total = 0
    n_correct_nocase = 0
    output = {'responses': []}
    for image_path in tqdm(image_list):
        gt_text, gt_image_bbox, google_response = compute_score(image_path, gt_data_path)
        if not google_response:
            #n_total += 1 #########
            pred_text = ''
            # continue
        else:
            pred_text = google_response[1].description # google_response[0].description always has a \n at the end
            google_bbox = get_google_bbox(google_response)
            tp = 0  # (ious.max(dim=1)[0] > 0.5).sum().item()
            fp = 0  # (ious.max(dim=0)[0] < 0.5).sum().item()
            fn = len(gt_image_bbox) - tp
        if pred_text.lower() == gt_text.lower():
            n_correct_nocase += 1

        #response_string = "\n".join(google_response)

        _, (s, i, d) = levenshtein(gt_text.lower(), pred_text.lower())
        cer_s += s
        cer_i += i
        cer_d += d
        cer_n += len(gt_text.lower())

        #ious = box_iou(torch.from_numpy(np.array(gt_image_bbox).astype('float')), torch.from_numpy(np.array(google_bbox).astype('float')))

        tp_total += tp
        fp_total += fp
        fn_total += fn
        n_total += 1
        #print('\n')
        #print('The GT text is :', gt_text)
        #print('The Pred text is :', pred_text)
        #print('\n')

        data = {}
        data['pred_text'] = pred_text
        data['gt_text'] = gt_text
        data['image_path'] = image_path

        output['responses'].append(data)

    e = 0.001
    prec = tp_total / (tp_total + fp_total + e)
    rec = tp_total / (tp_total + fn_total + e)

    case_insensitive_accuracy = n_correct_nocase / float(n_total)

    #print(prec, rec)
    CER = (cer_s + cer_i + cer_d) / float(cer_n)
    print('CER :', CER)
    print('WER :', 1 - case_insensitive_accuracy)

    output['CER'] = CER
    output['WER'] = 1 - case_insensitive_accuracy

    with open('data.json', 'w') as outfile:
        json.dump(output, outfile)


def main():
     path = '/Users/dnathawani/Desktop/hwt/formsA-D/a01-000u.png'
     response = detect_text_google_vision(path)
     print(response)


def page_image_benchmark():
    img_dir = '/Users/dnathawani/Desktop/hwt/formsI-Z'
    gt_data_file = '/Users/dnathawani/Desktop/hwt/ascii/words.txt'

    test_split_file = '/Users/dnathawani/Desktop/hwt/largeWriterIndependentTextLineRecognitionTask/testset.txt'
    test_indices = []
    with open(test_split_file, 'r') as f:
        for line in f:
            line = line.strip('\n')[:-3]
            test_indices.append(line)

    image_list = []
    for root, dirs, files in os.walk(img_dir):
        if not dirs:
            for file in files:
                file_id = file.split('.')[0]
                #file_id = file_id[:-1]
                #if file_id in test_indices:
                image_list.append(os.path.join(root, file))

    #image_list = image_list[:50]

    print(len(image_list))

    d_tp_total, d_fp_total, d_fn_total = 0., 0., 0.
    r_tp_total, r_fp_total, r_fn_total = 0., 0., 0.
    output = {'responses': []}

    label_path = os.path.join(gt_data_file)
    with open(label_path) as f:
        content = f.readlines()
    gt_dict = defaultdict(lambda: defaultdict(list))
    gt_dict_words = defaultdict(list)

    for gt in content:
        if gt.startswith('#'):
            continue
        line = gt.split()
        word_id = line[0]
        image_id = line[0][:-6]
        bbox = [int(line[3]), int(line[4]), int(line[5]) + int(line[3]), int(line[6]) + int(line[4])]
        gt_dict[image_id][word_id] = bbox
        gt_dict_words[image_id].append(line[8])

    x_min, y_min, x_max, y_max = 100000, 100000, 0, 0
    hw_rect_dict = defaultdict(list)
    for image_id, bbox_dict in gt_dict.items():
        bbox_list = list(gt_dict[image_id].values())
        for bbox in bbox_list:
            x_min = min(x_min, bbox[0])
            y_min = min(y_min, bbox[1])
            x_max = max(x_max, bbox[2])
            y_max = max(y_max, bbox[3])

        hw_rect_dict[image_id] = [x_min, y_min, x_max, y_max]
        x_min, y_min, x_max, y_max = 10000, 10000, 0, 0

    for image_path in tqdm(image_list):
        #'a04-006.png'
        image_id = os.path.split(image_path)[1].split('.')[0]
        gt_image_bbox = list(gt_dict[image_id].values())
        google_response = compute_score(image_path, gt_dict)
        if not google_response:
            pred_text = ''
        else:
            pred_text = google_response[1].description # google_response[0].description always has a \n at the end
            google_bbox, google_bbox_word_predictions = get_google_bbox(google_response, hw_rect_dict[image_id])
            ious = box_iou(torch.from_numpy(np.array(gt_image_bbox).astype('float')),
                           torch.from_numpy(np.array(google_bbox).astype('float')))
            torch.set_printoptions(edgeitems=100)
            #print(ious)
            #print(ious.max(dim=1))
            #print(ious.shape)
            #print(len(gt_image_bbox), len(gt_image_bbox[0]))
            ##print(len(google_bbox), len(google_bbox[0]))

            google_bbox.append(hw_rect_dict[image_id])

            plot_gt_pred_bbox(image_path, google_bbox, gt_image_bbox)

            values_1, indices_1 = ious.max(dim=1)
            tp_mask = values_1 > 0.5
            d_tp = (tp_mask).sum().item()

            r_tp_list = []
            #print('l1', len(gt_dict_words[image_id]))
            #print('l2', len(google_bbox_word_predictions))
            for i in range(len(gt_dict_words[image_id])):
                if gt_dict_words[image_id][i] == google_bbox_word_predictions[indices_1[i]]:
                    # Recognition True Positive
                    r_tp_list.append(i)
            r_tp = len(list(compress(r_tp_list, tp_mask)))
            r_fp = len(google_bbox_word_predictions) - r_tp
            #print('Recognition True Positive', r_tp)

            #d_tp = (ious.max(dim=1)[0] > 0.5).sum().item()
            d_fp = (ious.max(dim=0)[0] < 0.5).sum().item()
            d_fn = len(gt_image_bbox) - d_tp
            r_fn = len(gt_image_bbox) - r_tp

        d_tp_total += d_tp
        d_fp_total += d_fp
        d_fn_total += d_fn

        r_tp_total += r_tp
        r_fp_total += r_fp
        r_fn_total += r_fn
        #print('\n')
        #print('The GT text is :', gt_text)
        #print('The Pred text is :', pred_text)
        #print('\n')

        data = {}
        #data['pred_text'] = pred_text
        #data['gt_text'] = gt_text
        data['image_path'] = image_path
        output['responses'].append(data)

    e = 0.000001
    detection_precision = d_tp_total / (d_tp_total + d_fp_total + e)
    detection_recall = d_tp_total / (d_tp_total + d_fn_total + e)

    recognition_precision = r_tp_total / (r_tp_total + r_fp_total + e)
    recognition_recall = r_tp_total / (r_tp_total + r_fn_total + e)

    print('Detection Precision :', detection_precision)
    print('Detection Recall :', detection_recall)

    print('Recognition Precision :', recognition_precision)
    print('Recognition Recall :', recognition_recall)

    output['detection_precision'] = detection_precision
    output['detection_recall'] = detection_recall

    output['recognition_precision'] = recognition_precision
    output['recognition_recall'] = recognition_recall

    with open('page_level_benchmark.json', 'w') as outfile:
        json.dump(output, outfile)


if __name__ == "__main__":
    #main()
    #word_image_benchmarking()
    page_image_benchmark()
