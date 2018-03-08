#encoding=utf-8
import os
import time
import sys
import base64
import math
import random

import urllib.request  
import datetime
import logging
import optparse
import tornado.wsgi
import tornado.httpserver
import numpy as np
import pandas as pd

from flask import Flask, request, redirect, url_for,render_template
from werkzeug import secure_filename
from werkzeug import SharedDataMiddleware

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf
import cv2
from nets import ssd_vgg_300, ssd_common, np_methods
from preprocessing import ssd_vgg_preprocessing
import visualization
from datasets import pascalvoc_common 
slim = tf.contrib.slim


base_dir = os.getcwd()
UPLOAD_FOLDER = base_dir + '/demo/upload'
DETECTED_FOLDER = base_dir +  '/demo/detected'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['DETECTED_FOLDER'] = DETECTED_FOLDER



# Main image processing routine.
def process_image(img, params, select_threshold=0.5, nms_threshold=.45, net_shape=(300, 300)):
    # Run SSD network
    isess = params['isess']
    image_4d = params['image_4d']
    predictions = params['predictions']
    localisations = params['localisations']
    bbox_img = params['bbox_img']
    img_input = params['img_input']
    ssd_anchors = params['ssd_anchors']

    rimg, rpredictions, rlocalisations, rbbox_img = isess.run([image_4d, predictions, localisations, bbox_img],
                                                              feed_dict={img_input: img})
    
    # Get classes and bboxes from the net outputs.
    rclasses, rscores, rbboxes = np_methods.ssd_bboxes_select(
            rpredictions, rlocalisations, ssd_anchors,
            select_threshold=select_threshold, img_shape=net_shape, num_classes=21, decode=True)
    
    rbboxes = np_methods.bboxes_clip(rbbox_img, rbboxes)
    rclasses, rscores, rbboxes = np_methods.bboxes_sort(rclasses, rscores, rbboxes, top_k=400)
    rclasses, rscores, rbboxes = np_methods.bboxes_nms(rclasses, rscores, rbboxes, nms_threshold=nms_threshold)
    # Resize bboxes to original image shape. Note: useless for Resize.WARP!
    rbboxes = np_methods.bboxes_resize(rbbox_img, rbboxes)
    return rclasses, rscores, rbboxes


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    return render_template('index.html', has_result=False)


@app.route('/classify_url', methods=['GET'])
def classify_url():
    imageurl = request.args.get('imageurl', '')
    try:
        # download
        user_Agent = 'Mozilla/5.0 (Windows NT 6.2) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/55.0.2883.87 Safari/537.36'
        header = {'User-Agent':user_Agent}
        req = urllib.request.Request(imageurl,headers=header)
        raw_data = urllib.request.urlopen(req).read()

        filename = os.path.join(UPLOAD_FOLDER, 'tmp.jpg')
        with open(filename,'wb') as f:
            f.write(raw_data)

    except Exception as err:
        # For any exception we encounter in reading the image, we will just
        # not continue.
        logging.info('URL Image open error: %s', err)
        return render_template(
            'index.html', has_result=True,
            result=(False, 'Cannot open image from URL.')
        )

    logging.info('Image: %s', imageurl)

    results,new_img_base64 = app.clf.classify_image(filename)
    return render_template(
        'index.html', has_result=True, result=results, imagesrc=new_img_base64)


@app.route('/classify_upload', methods=['POST'])
def classify_upload():
    try: 
        app.logger.addHandler(logging.StreamHandler(sys.stdout))
        app.logger.setLevel(logging.DEBUG)

        # We will save the file to disk for possible data collection.
        imagefile = request.files['imagefile']
        app.logger.debug(imagefile)
        sys.stdout.write(imagefile.filename+'*****'+'\n')
        if imagefile and allowed_file(imagefile.filename):
            filename_ = str(datetime.datetime.now()).replace(' ', '_') + secure_filename(imagefile.filename)
            sys.stdout.write('fkfkfkfk'+ filename_+ '\n')
            filename = os.path.join(UPLOAD_FOLDER, filename_)
            imagefile.save(filename)

    except Exception as err:
        #logging.info('Uploaded image open error: %s', err)
        return render_template(
            'index.html', has_result=True,
            result=(False, 'Cannot open uploaded image.')
        )
    
    sys.stdout.write( str(imagefile.filename) + '1111111111111111111111111' +'\n')
    results,new_img_base64 = app.clf.classify_image(filename)

    return render_template(
        'index.html', has_result=True, result=results,
        imagesrc=new_img_base64
    )


def embed_image_html(fileps):
    """Creates an image embedded in HTML base64 format."""

    with open(fileps, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())

    return 'data:image/jpg;base64,' + str(encoded_string)[2:-1]


def init_model(ckpt_path):

    l_VOC_CLASS = [name for name, tup in pascalvoc_common.VOC_LABELS.items()]

    # TensorFlow session: grow memory when needed. TF, DO NOT USE ALL MY GPU MEMORY!!!
    gpu_options = tf.GPUOptions(allow_growth=True)
    config = tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options)
    isess = tf.InteractiveSession(config=config)
    
    # Input placeholder.
    net_shape = (300, 300)
    data_format = 'NHWC'
    img_input = tf.placeholder(tf.uint8, shape=(None, None, 3))
    
    # Evaluation pre-processing: resize to SSD net shape.
    image_pre, labels_pre, bboxes_pre, bbox_img = ssd_vgg_preprocessing.preprocess_for_eval(img_input, None, None, net_shape, data_format, resize=ssd_vgg_preprocessing.Resize.WARP_RESIZE)
    image_4d = tf.expand_dims(image_pre, 0)

    
    # Define the SSD model.
    reuse = True if 'ssd_net' in locals() else None
    ssd_net = ssd_vgg_300.SSDNet()
    with slim.arg_scope(ssd_net.arg_scope(data_format=data_format)):
        predictions, localisations, _, _ = ssd_net.net(image_4d, is_training=False, reuse=reuse)

    # Restore SSD model.
    ckpt_filename = ckpt_path
    # ckpt_filename = '../checkpoints/VGG_VOC0712_SSD_300x300_ft_iter_120000.ckpt'
    isess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(isess, ckpt_filename)

    # SSD default anchor boxes.
    ssd_anchors = ssd_net.anchors(net_shape)

    return {'l_VOC_CLASS':l_VOC_CLASS,'ssd_anchors':ssd_anchors,'img_input':img_input,'isess':isess,'image_4d':image_4d,'predictions':predictions,'localisations':localisations,'bbox_img':bbox_img}


class ImagenetClassifier(object):

    # 预先加载模型
    def __init__(self, l_VOC_CLASS, ssd_anchors, img_input, isess, image_4d, predictions, localisations, bbox_img):
        logging.info('Loading net and associated files...')
        self.l_VOC_CLASS = l_VOC_CLASS
        self.ssd_anchors = ssd_anchors
        self.img_input = img_input
        self.isess = isess
        self.image_4d = image_4d
        self.predictions = predictions
        self.localisations = localisations
        self.bbox_img = bbox_img

    def classify_image(self, filename):
        try:
            params = {'ssd_anchors':self.ssd_anchors,'img_input':self.img_input,'isess':self.isess,'image_4d':self.image_4d,'predictions':self.predictions,'localisations':self.localisations,'bbox_img':self.bbox_img}

            # read image
            img = mpimg.imread(filename)


            starttime = time.time()
            rclasses, rscores, rbboxes =  process_image(img,params)
            visualization.bboxes_draw_on_img(img, rclasses, rscores, rbboxes, visualization.colors_plasma)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            endtime = time.time()

            bet_result = [(str(idx+1)+' : '+ self.l_VOC_CLASS[v], '%.5f' % rscores[idx]) for idx, v in enumerate(rclasses)]


            # save image after draw box
            fileout = str(datetime.datetime.now()).replace(' ', '_') + 'processed_img' + '.jpg' 
            fileps = os.path.join(DETECTED_FOLDER, fileout)
            cv2.imwrite(fileps,img)
            sys.stdout.write( '-------------------------'+'\n')

            sys.stdout.write( 'dddddddddddddd'+'\n')
            sys.stdout.write( fileps + '\n')
            sys.stdout.write( 'dddddddddddddd'+'\n')

            new_img_base64 = embed_image_html(fileps)


            rtn = (True, (rclasses, rscores, rbboxes), bet_result, '%.3f' % (endtime - starttime))
            return rtn,new_img_base64

        except Exception as err:
            logging.info('Classification error: %s', err)
            return (False, 'Something went wrong when classifying the '
                           'image. Maybe try another one?')



def start_tornado(app, port=5000):
    http_server = tornado.httpserver.HTTPServer(
        tornado.wsgi.WSGIContainer(app))
    http_server.listen(port)
    print("Tornado server starting on port {}".format(port))
    tornado.ioloop.IOLoop.instance().start()


def start_from_terminal(app):
    """
    Parse command line options and start the server.
    """
    parser = optparse.OptionParser()
    parser.add_option(
        '-d', '--debug',
        help="enable debug mode",
        action="store_true", default=False)
    parser.add_option(
        '-p', '--port',
        help="which port to serve content on",
        type='int', default=5000)
    parser.add_option(
        '-g', '--gpu',
        help="use gpu mode",
        action='store_true', default=True)

    opts, args = parser.parse_args()
    # ImagenetClassifier.default_args.update({'gpu_mode': opts.gpu})

    # Initialize classifier + warm start by forward for allocation
    ckpt_path = '/home/gl/objdet_web/checkpoints/ssd_300_vgg.ckpt'
    init_stateModel = init_model(ckpt_path)
    app.clf = ImagenetClassifier(**init_stateModel)

    if opts.debug:
        app.run(debug=True, host='0.0.0.0', port=opts.port)
    else:
        start_tornado(app, opts.port)


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    if not os.path.exists(DETECTED_FOLDER):
        os.makedirs(DETECTED_FOLDER)
    start_from_terminal(app)
