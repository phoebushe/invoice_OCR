from __future__ import print_function

import cv2
import time
from PIL import Image
import glob
import os
import os.path as osp
import shutil
import sys
import numpy as np
import tensorflow as tf
from ctpn.lib.fast_rcnn.config import cfg
from tensorflow.python.platform import gfile
from ctpn.lib.fast_rcnn.test import _get_blobs
from ctpn.lib.rpn_msr.proposal_layer_tf import proposal_layer
from ctpn.lib.text_connector.detectors import TextDetector

# crnn packages
import torch
from torch.autograd import Variable
import utils
import dataset
from PIL import Image
import models.crnn as crnn
import alphabets
str1 = alphabets.alphabet

crnn_model_path = 'models/mixed_second_finetune_acc97p7.pth'
alphabet = str1
nclass = len(alphabet)+1




sys.path.append(os.getcwd())

ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__)))
DATA_DIR = osp.abspath(osp.join(ROOT_DIR, 'data'))
PIXEL_MEANS = np.array([[[102.9801, 115.9465, 122.7717]]])
SCALES = (600,)
MAX_SIZE = 1000


def resize_im(im, scale, max_scale=None):
    f = float(scale) / min(im.shape[0], im.shape[1])
    if max_scale != None and f * max(im.shape[0], im.shape[1]) > max_scale:
        f = float(max_scale) / max(im.shape[0], im.shape[1])
    return cv2.resize(im, None, None, fx=f, fy=f, interpolation=cv2.INTER_CUBIC), f


def generate_detect_box(img, image_name, boxes, scale):
    dict = []
    base_name = image_name.split('/')[-1]
    base_name = base_name.split('.')[0]
    img = cv2.resize(img, None, None, fx=1.0 / scale, fy=1.0 / scale, interpolation=cv2.INTER_CUBIC)
    with open('data/results/' + 'res_{}.txt'.format(base_name.split('.')[0]), 'w') as f:
        for i in range(len(boxes)):
            box = boxes[i]
            min_x = min(int(box[0] / scale), int(box[2] / scale), int(box[4] / scale), int(box[6] / scale))
            min_y = min(int(box[1] / scale), int(box[3] / scale), int(box[5] / scale), int(box[7] / scale))
            max_x = max(int(box[0] / scale), int(box[2] / scale), int(box[4] / scale), int(box[6] / scale))
            max_y = max(int(box[1] / scale), int(box[3] / scale), int(box[5] / scale), int(box[7] / scale))

            line = ','.join([str(min_x), str(min_y), str(max_x), str(max_y)]) + '\r\n'
            f.write(line)
    f.close()
    with open('data/results/' + 'res_{}.txt'.format(base_name.split('.')[0]), 'r') as f2:
        lines = f2.readlines()
        if lines != []:
            for line in lines:
                line = line.strip('\n')
                temp = line.split(',')
                temp = list(map(int, temp))
                dict.append(temp)
            array = np.array(dict)
            results = array[array[:, 1].argsort()]
            for j in range(len(results)):
                result = results[j]
                minx = result[0]
                miny = result[1]
                maxx = result[2]
                maxy = result[3]

                # if (minx -2) > 0:
                #     minx = minx - 1
                # if (miny -2) > 0:
                #     miny = miny - 1



                temp = img[miny: maxy, minx: maxx]

                h, w = temp.shape[:2]

                # if h <= 20 :
                #     scale2 = h / 20
                #     temp = cv2.resize(temp, (round(w / scale2), 20), interpolation=cv2.INTER_CUBIC)



                if base_name == 'img_amount':
                    temp = temp
                elif base_name == 'img_price':
                    if (minx -2) > 0:
                        minx = minx - 1
                    if (miny -2) > 0:
                        miny = miny - 1
                    temp = img[miny: maxy, minx: maxx]
                elif base_name == 'img_taxes':
                    if (minx -2) > 0:
                        minx = minx - 1
                    if (miny -2) > 0:
                        miny = miny - 1
                    temp = img[miny: maxy, minx: maxx]
                elif base_name == 'img_buyer' and j == 1:
                    temp = temp
                elif base_name == 'img_seller' and j == 1:
                    temp = temp
                else:
                    scale2 = h / 32
                    temp = cv2.resize(temp, (round(w / scale2), 32), interpolation=cv2.INTER_CUBIC)

                # kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)
                # dst = cv2.filter2D(temp, -1, kernel=kernel)
                # lap_9 = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
                #
                # dst = cv2.filter2D(temp, cv2.CV_8U, lap_9)
                # dst = cv2.fastNlMeansDenoisingColored(dst, None, 3, 3, 7, 21)

                cv2.imwrite('data/after_detect/' + base_name + '/'+ str(j) + '.jpg', temp)



def sort_box():
    path = 'data/results/'
    filenames = os.listdir(path)

    for filename in filenames:
        dict = []
        with open(path + filename, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip('\n')
                temp = line.split(',')
                temp = list(map(int, temp))
                dict.append(temp)
            array = np.array(dict)
            results = array[array[:, 1].argsort()]
            with open(path + filename, 'w') as f:
                for result in results:
                    sort_result = ','.join([str(result[0]), str(result[1]), str(result[2]), str(result[3])]) + '\r\n'
                    f.write(sort_result)



# crnn文本信息识别
def crnn_recognition(cropped_image, model):
    converter = utils.strLabelConverter(alphabet)

    image = cropped_image.convert('L')

    ##
    w = int(image.size[0] / (280 * 1.0 / 160))
    transformer = dataset.resizeNormalize((w, 32))
    image = transformer(image)
    if torch.cuda.is_available():
        image = image.cuda()
    image = image.view(1, *image.size())
    image = Variable(image)

    model.eval()
    preds = model(image)

    _, preds = preds.max(2)
    preds = preds.transpose(1, 0).contiguous().view(-1)

    preds_size = Variable(torch.IntTensor([preds.size(0)]))
    sim_pred = converter.decode(preds.data, preds_size.data, raw=False)
    return sim_pred


def crop_invoice(image, image2):
    gray_src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    gray_src = cv2.bitwise_not(gray_src)
    binary_src = cv2.adaptiveThreshold(gray_src, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -2)
    # 提取水平线
    hline = cv2.getStructuringElement(cv2.MORPH_RECT, (int((src.shape[1] / 3)), 1), (-1, -1))
    # 提取垂直线
    vline = cv2.getStructuringElement(cv2.MORPH_RECT, (1, int((src.shape[0] / 3))), (-1, -1))
    dst_vline = cv2.morphologyEx(binary_src, cv2.MORPH_OPEN, vline)
    dst_vline = cv2.bitwise_not(dst_vline)
    dst_hline = cv2.morphologyEx(binary_src, cv2.MORPH_OPEN, hline)
    dst_hline = cv2.bitwise_not(dst_hline)

    dst = dst_hline + dst_vline
    dst2 = cv2.bitwise_and(dst_vline, dst_hline)
    point_x = []
    point_y = []
    for i in range(dst.shape[0]):
        for j in range(dst.shape[1]):
            if dst[i][j] == 0:
                point_x.append(j)
                point_y.append(i)

    point_x_max = max(point_x) + 5
    point_y_max = max(point_y) + 5
    point_x_min = min(point_x) - 5
    point_y_min = min(point_y) - 5


    box = (point_x_min, point_y_min, point_x_max, point_y_max)  # 最大框box
    cropImg_Bigbox = im.crop(box)  # 最大框裁剪
    w, h = cropImg_Bigbox.size

    ##购买方的框##
    box_buyer = (point_x_min + 0.16 * w, point_y_min, point_x_min + 0.575 * w, point_y_min + 0.24 * h)
    img_buyer = im.crop(box_buyer)
    img_buyer.save('data/demo/img_buyer.jpg', 'jpeg')

    ##货物或服务名称##
    box_name = (point_x_min + 0.002 * w, point_y_min + 0.278 * h, point_x_min + 0.26 * w, point_y_min + 0.65 * h)
    img_name = im.crop(box_name)
    img_name.save('data/demo/img_name.jpg', 'jpeg')


    ##规格型号##
    box_specification = (
    point_x_min + 0.261 * w, point_y_min + 0.278 * h, point_x_min + 0.388 * w, point_y_min + 0.65 * h)
    img_specification = im.crop(box_specification)
    img_specification.save('data/demo/img_specification.jpg', 'jpeg')

    ##单位##
    box_unit = (point_x_min + 0.388 * w, point_y_min + 0.276 * h, point_x_min + 0.447 * w, point_y_min + 0.65 * h)
    img_unit = im.crop(box_unit)
    img_unit.save('data/demo/img_unit.jpg', 'jpeg')


    ##数量##
    box_number = (point_x_min + 0.449 * w, point_y_min + 0.276 * h, point_x_min + 0.548 * w, point_y_min + 0.65 * h)
    img_number = im.crop(box_number)
    img_number.save('data/demo/img_number.jpg', 'jpeg')

    ##单价##
    box_unit_price = (point_x_min + 0.549 * w, point_y_min + 0.276 * h, point_x_min + 0.648 * w, point_y_min + 0.65 * h)
    img_unit_price = im.crop(box_unit_price)
    img_unit_price.save('data/demo/img_unit_price.jpg', 'jpeg')

    ##金额##
    box_price = (point_x_min + 0.649 * w, point_y_min + 0.610 * h, point_x_min + 0.797 * w, point_y_min + 0.695 * h)
    img_price = im.crop(box_price)
    img_price.save('data/demo/img_price.jpg', 'jpeg')

    ##税额##
    box_taxes = (point_x_min + 0.850 * w, point_y_min + 0.610 * h, point_x_min + 0.995 * w, point_y_min + 0.695 * h)
    img_taxes = im.crop(box_taxes)
    img_taxes.save('data/demo/img_taxes.jpg', 'jpeg')

    ##价税合计##
    box_amount = (point_x_min + 0.778 * w, point_y_min + 0.696 * h, point_x_min + 0.995 * w, point_y_min + 0.780 * h)
    img_amount = im.crop(box_amount)
    img_amount.save('data/demo/img_amount.jpg', 'jpeg')

    ##销售方的框##
    box_seller = (point_x_min + 0.16 * w, point_y_min + 0.765 * h, point_x_min + 0.575 * w, point_y_min + 0.990 * h)
    img_seller = im.crop(box_seller)
    img_seller.save('data/demo/img_seller.jpg', 'jpeg')

    return img_buyer, img_name, img_specification, img_unit, img_number, \
           img_unit_price, img_price, img_taxes, img_amount, img_seller

def check_path():
    if os.path.exists("data/results/"):
        shutil.rmtree("data/results/")
    os.makedirs("data/results/")

    lists = ['img_buyer', 'img_name', 'img_specification', 'img_unit', 'img_number', \
           'img_unit_price', 'img_price', 'img_taxes', 'img_amount', 'img_seller']

    for list in lists:
        if os.path.exists("data/after_detect/" + list):
            shutil.rmtree("data/after_detect/" + list)
        os.makedirs("data/after_detect/" + list)




if __name__ == '__main__':

    ##读取图片##
    time1 = time.time()
    pic = '2.jpeg'
    im = Image.open(pic)
    src = cv2.imread(pic)

    ##将增值税发票需要识别的区域划分出来##
    img_buyer, img_name, img_specification, img_unit, \
    img_number, img_unit_price, img_price, img_taxes, \
    img_amount, img_seller = crop_invoice(im, src)
    time2 = time.time()
    print("read pic", time2 - time1)

    ##CTPN文字检测##
    check_path()
    time3 = time.time()
    # init session
    config = tf.ConfigProto(allow_soft_placement=True)  #允许tf自动选择一个存在并且可用的设备来运行操作。
    sess = tf.Session(config=config)
    with gfile.FastGFile('ctpn/ctpn_1.pb', 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        sess.graph.as_default()
        tf.import_graph_def(graph_def, name='')
    sess.run(tf.global_variables_initializer())

    input_img = sess.graph.get_tensor_by_name('Placeholder:0')
    output_cls_prob = sess.graph.get_tensor_by_name('Reshape_2:0')
    output_box_pred = sess.graph.get_tensor_by_name('rpn_bbox_pred/Reshape_1:0')
    time4 = time.time()
    print("sess time", time4 - time3)


    im_names = glob.glob(os.path.join(DATA_DIR, 'demo', '*.png')) + \
               glob.glob(os.path.join(DATA_DIR, 'demo', '*.jpg'))  #匹配所有的符合条件的文件，并将其以list的形式返回

    time5 = time.time()
    for im_name in im_names:
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print(('Demo for {:s}'.format(im_name)))
        img = cv2.imread(im_name)
        img, scale = resize_im(img, scale=600, max_scale=1200) ##缩放至合适比例##
        blobs, im_scales = _get_blobs(img, None)  ##得到输入图片和缩放系数##
        if True:
            im_blob = blobs['data']
            blobs['im_info'] = np.array(
                [[im_blob.shape[1], im_blob.shape[2], im_scales[0]]],
                dtype=np.float32)
        cls_prob, box_pred = sess.run([output_cls_prob, output_box_pred], feed_dict={input_img: blobs['data']})
        rois, _ = proposal_layer(cls_prob, box_pred, blobs['im_info'], 'TEST', anchor_scales=cfg.ANCHOR_SCALES)

        scores = rois[:, 0]
        boxes = rois[:, 1:5] / im_scales[0]
        textdetector = TextDetector()
        boxes = textdetector.detect(boxes, scores[:, np.newaxis], img.shape[:2])
        generate_detect_box(img, im_name, boxes, scale)
        #sort_box()


    time6 = time.time()
    print("text detect", time6 - time5)

    time7 = time.time()


