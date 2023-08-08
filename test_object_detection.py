import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import pathlib
import cv2
import math
from PIL import Image
from google.protobuf import text_format
import platform
import os


#print('Python version:', platform.python_version())
#print('Tensorflow version:', tf.__version__)
#print('Keras version:', tf.keras.__version__)


# Create cache folder.
#os.system("mkdir ./tmp")
PROTOC_ZIP="protoc-23.4-linux-x86_64.zip"

#os.system("""curl -OL https://github.com/protocolbuffers/protobuf/releases/download/v23.4/protoc-23.4-linux-x86_64.zip""")
#os.system("""unzip -o {} -d ./tmp/protoc""".format(PROTOC_ZIP))
#os.system("""rm -f {}""".format(PROTOC_ZIP))
os.system("""./tmp/protoc/bin/protoc ./protos/*.proto --python_out=.""")
from protos import string_int_label_map_pb2
LABELS_NAME = 'mscoco_label_map.pbtxt'


# Loads the module from internet, unpacks it and initializes a Tensorflow saved model.
def load_model(model_name):
    model_url = 'http://download.tensorflow.org/models/object_detection/' + model_name + '.tar.gz'

    model_dir = tf.keras.utils.get_file(
        fname=model_name,
        origin=model_url,
        untar=True,
        cache_dir=pathlib.Path('.tmp').absolute()
    )
    model = tf.saved_model.load(model_dir + '/saved_model')

    return model


MODEL_NAME = 'ssdlite_mobilenet_v2_coco_2018_05_09'
saved_model = load_model(MODEL_NAME)
# Loading default model signature.
model = saved_model.signatures['serving_default']


def load_labels(labels_name):
    labels_url = 'https://raw.githubusercontent.com/tensorflow/models/master/research/object_detection/data/' + labels_name

    labels_path = tf.keras.utils.get_file(
        fname=labels_name,
        origin=labels_url,
        cache_dir=pathlib.Path('.tmp').absolute()
    )

    labels_file = open(labels_path, 'r')
    labels_string = labels_file.read()

    labels_map = string_int_label_map_pb2.StringIntLabelMap()
    try:
        text_format.Merge(labels_string, labels_map)
    except text_format.ParseError:
        print("ERROR")
        labels_map.ParseFromString(labels_string)

    labels_dict = {}
    for item in labels_map.item:
        labels_dict[item.id] = item.display_name
    print("label", labels_dict)
    return labels_dict


def display_image(image_np):
    plt.figure()
    plt.imshow(image_np)


def detect_objects_on_image(image, model):
    image = np.asarray(image)

    input_tensor = tf.convert_to_tensor(image)
    # Adding one more dimension since model expect a batch of images.
    input_tensor = input_tensor[tf.newaxis, ...]
    print(input_tensor)
    output_dict = model(input_tensor)

    num_detections = int(output_dict['num_detections'])
    output_dict = {
        key:value[0, :num_detections].numpy()
        for key,value in output_dict.items()
        if key != 'num_detections'
    }
    output_dict['num_detections'] = num_detections
    output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)

    return output_dict




def draw_detections_on_image(image, detections, labels):
    image_with_detections = image
    width, height, channels = image_with_detections.shape

    font = cv2.FONT_HERSHEY_SIMPLEX
    color = (0, 255, 0)
    label_padding = 5

    num_detections = detections['num_detections']
    if num_detections > 0:
        for detection_index in range(num_detections):
            detection_score = detections['detection_scores'][detection_index]
            detection_box = detections['detection_boxes'][detection_index]
            detection_class = detections['detection_classes'][detection_index]
            detection_label = labels[detection_class]
            detection_label_full = detection_label + ' ' + str(math.floor(100 * detection_score)) + '%'

            y1 = int(width * detection_box[0])
            x1 = int(height * detection_box[1])
            y2 = int(width * detection_box[2])
            x2 = int(height * detection_box[3])

            # Detection rectangle.
            image_with_detections = cv2.rectangle(
                image_with_detections,
                (x1, y1),
                (x2, y2),
                color,
                3
            )

            # Label background.
            label_size = cv2.getTextSize(
                detection_label_full,
                cv2.FONT_HERSHEY_COMPLEX,
                0.7,
                2
            )
            image_with_detections = cv2.rectangle(
                image_with_detections,
                (x1, y1 - label_size[0][1] - 2 * label_padding),
                (x1 + label_size[0][0] + 2 * label_padding, y1),
                color,
                -1
            )

            # Label text.
            cv2.putText(
                image_with_detections,
                detection_label_full,
                (x1 + label_padding, y1 - label_padding),
                font,
                0.7,
                (0, 0, 0),
                1,
                cv2.LINE_AA
            )

    return image_with_detections

labels = load_labels(LABELS_NAME)
image_np = np.array(Image.open("dog.jpeg"))
detections = detect_objects_on_image(image_np, model)
image_with_detections = draw_detections_on_image(image_np, detections, labels)
img = Image.fromarray(image_with_detections, "RGB")
image_filename = "opengenus_image.jpeg"
img.save(image_filename)
