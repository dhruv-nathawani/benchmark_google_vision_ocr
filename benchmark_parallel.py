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
import threading
import sys
import time
from concurrent.futures import ThreadPoolExecutor, wait, as_completed


def detect_text_google_vision(path, client):
    """Detects text in the file."""
    with io.open(path, 'rb') as image_file:
        content = image_file.read()
    image = vision.types.Image(content=content)
    response = client.text_detection(image=image)
    return response.text_annotations


def get_rectangle_from_polygon(polygon):
    'Input format[x, y, x, y..]'
    x = polygon[::2]
    y = polygon[1::2]
    return [min(x), min(y), max(x), max(y)]


def get_google_bbox(google_response):
    google_bbox = []
    for result in google_response[1:]:
        polygon = []
        for vertex in result.bounding_poly.vertices:
            polygon.append(vertex.x)
            polygon.append(vertex.y)
        google_bbox.append(get_rectangle_from_polygon(polygon))
    return google_bbox


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

def main():
    img_dir = '/Users/dnathawani/Desktop/hwt/words'
    gt_data_path = '/Users/dnathawani/Desktop/hwt/xml'
    client = vision.ImageAnnotatorClient()
    thread_count = 7

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

    #image_list = image_list[:5]
    print("The length of image_list is :", len(image_list))
    futures = []

    tp_total, fp_total, fn_total = 0., 0., 0.
    cer_s = 0
    cer_i = 0
    cer_d = 0
    cer_n = 0
    n_total = 0
    n_correct_nocase = 0
    output = {'responses': []}

    number_image_submitted = 0 #submitted to threadpool executor
    number_image_processed = 0
    start_time = time.time()
    with ThreadPoolExecutor(thread_count) as executor:

        for image_path in tqdm(image_list):
            try:
                queue_size = executor._work_queue.qsize()
                if queue_size < 18000:  # do not flood the executor's queue
                    futures.append(executor.submit(compute_score, image_path, gt_data_path, client))
                    number_image_submitted += 1
            except Exception as e:
                print(e)
                continue


    for completed_future in as_completed(futures):
        number_image_processed += 1
        gt_text, gt_image_bbox, google_response = completed_future.result()
        #print("Printing response", google_response)
        #print("Printing GT text", gt_text)
        if not google_response:
            pred_text = ''
        else:
            pred_text = google_response[1].description
            google_bbox = get_google_bbox(google_response)

            tp = 0  # (ious.max(dim=1)[0] > 0.5).sum().item()
            fp = 0  # (ious.max(dim=0)[0] < 0.5).sum().item()
            fn = len(gt_image_bbox) - tp

            if pred_text.lower() == gt_text.lower():
                n_correct_nocase += 1

        _, (s, i, d) = levenshtein(gt_text.lower(), pred_text.lower())
        cer_s += s
        cer_i += i
        cer_d += d
        cer_n += len(gt_text.lower())

        #ious = box_iou(torch.from_numpy(np.array(gt_image_bbox).astype('float')), torch.from_numpy(np.array(google_bbox).astype('float')))

        tp_total += 0
        fp_total += 0
        fn_total += 0
        n_total += 1

        data = {}
        data['pred_text'] = pred_text
        data['gt_text'] = gt_text
        data['image_path'] = image_path

        output['responses'].append(data)

        with open('word_image_benchmark.json', 'w') as outfile:
            json.dump(output, outfile)

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

    with open('word_image_benchmark.json', 'w') as outfile:
        json.dump(output, outfile)

    elapsed_time = time.time() - start_time
    print("elapsed time = {}".format(elapsed_time))
    print("no of images submitted = {}".format(number_image_submitted))
    print("no of images processed = {}".format(number_image_processed))



def compute_score(image_path, gt_data_path, client):
    thread_name = threading.currentThread().getName()
    print("Thread [{}] is processing image = {}".format(thread_name, image_path))

    image_name = os.path.split(image_path)[1]
    xml_file = '-'.join(image_name.split('.')[0].split('-')[:-2])
    xml_file = os.path.join(gt_data_path, xml_file) + '.xml'
    root = ET.parse(xml_file).getroot()

    for type_tag in root.findall('handwritten-part/line/word'):
        if type_tag.get('id') == image_name.split('.')[0]:
            gt_text = type_tag.get('text')
            gt_image_bbox = type_tag.findall('cmp')
            for t in type_tag.findall('cmp'):
                gt_image_bbox = [t.get('x'), t.get('y'), t.get('width'), t.get('height')]

    google_response = detect_text_google_vision(image_path, client)
    if not google_response:
        return gt_text, 0, 0

    return gt_text, gt_image_bbox, google_response


if __name__ == "__main__":
    main()