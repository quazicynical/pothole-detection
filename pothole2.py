import cv2
import matplotlib.pyplot as plt
import urllib.request
import numpy as np
import concurrent.futures
import cv2
import tensorflow.compat.v1 as tf
import visualization_utils as vis_util

tf.disable_v2_behavior()
PATH_TO_CKPT = "/Users/niruphans/PycharmProjects/pothole_detec2/.venv/frozen_inference_graph.pb"
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess = tf.Session(graph=detection_graph)

image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
num_detections = detection_graph.get_tensor_by_name('num_detections:0')

category_index = {1: {'name': "pothole"}, 2: {'name': "no"}, 3: {'name': "no2"}, 4: {'name': "no3"}}

url = 'http://192.168.10.162/cam-hi.jpg'
im = None


def run1():
    cv2.namedWindow("live transmission", cv2.WINDOW_AUTOSIZE)
    while True:
        img_resp = urllib.request.urlopen(url)
        imgnp = np.array(bytearray(img_resp.read()), dtype=np.uint8)

        frame = cv2.imdecode(imgnp, -1)
        frame_expand = np.expand_dims(frame, axis=0)

        (boxes, scores, classes, num) = sess.run(
            [detection_boxes, detection_scores, detection_classes, num_detections],
            feed_dict={image_tensor: frame_expand})
        print(boxes)

        vis_util.visualize_boxes_and_labels_on_image_array(
            frame,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            category_index,
            use_normalized_coordinates=True,
            line_thickness=8,
            min_score_thresh=0.60)
        cv2.imshow('frame', frame)

        key = cv2.waitKey(5)
        if key == ord('q'):
            break

    cv2.destroyAllWindows()


if __name__ == '__main__':
    print("started")
    with concurrent.futures.ProcessPoolExecutor() as executer:
        f1 = executer.submit(run1)
