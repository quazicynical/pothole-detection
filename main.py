# import tensorflow as tf
import cv2
import numpy as np
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

vid = cv2.VideoCapture(1)
category_index = {1: {'name': "pothole"}, 2: {'name': "no"}, 3: {'name': "no2"}, 4: {'name': "no3"}}
while True:
    ret, frame = vid.read()

    # Display the resulting frame

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

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
vid.release()
cv2.destroyAllWindows()
