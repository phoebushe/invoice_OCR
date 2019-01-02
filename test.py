import numpy as np
import sys, os
import time

sys.path.append(os.getcwd())


# crnn packages
import torch
from torch.autograd import Variable
import utils
import dataset
from PIL import Image
import models.crnn as crnn
import alphabets
str1 = alphabets.alphabet

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--images_path', type=str, default='test_images/2.jpg', help='the path to your images')
opt = parser.parse_args()


# crnn params
# 3p6m_third_ac97p8.pth
crnn_model_path = 'models/crnn_Rec_done_1_28125.pth'
alphabet = str1
nclass = len(alphabet)+1


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



if __name__ == '__main__':

	# crnn network
    model = crnn.CRNN(32, 1, nclass, 256)
    if torch.cuda.is_available():
        model = model.cuda()
    print('loading pretrained model from {0}'.format(crnn_model_path))
    # 导入已经训练好的crnn模型
    model.load_state_dict(torch.load(crnn_model_path))
    
    started = time.time()
    ## read an image
    path = 'data/after_detect/'
    filenames = os.listdir(path)
    for filename in filenames:
        images_path = os.path.join(path, filename)
        images = os.listdir(images_path)
        for image in images:
            pic_path = path + filename + str('/') + image
            pic = Image.open(pic_path)
            result = crnn_recognition(pic, model)
            if (filename == 'img_amount') or (filename == 'img_taxes') or (filename == 'img_price'):
                result = result.replace(',', '.')
                correct_char = "0123456789."
                for c in result:
                    if not c in correct_char:
                        result = result.replace(c, '')

            finished = time.time()

            print('image:%s, result:%s' % (pic_path, result))

    
