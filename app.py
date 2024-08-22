import io
import tensorflow as tf
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from flask import Flask, request, jsonify
from PIL import Image
from src.data_structures import Point, Rect
from src.iso_standard import PhotographicRequirements


# Constants
IMAGE_SIZE = (160, 160)
MODEL_FACE_DETECTOR = "resources/models/face_detector.pb"
MODEL_ICAONET = "resources/models/icaonet.h5"

# Initialize the Flask application
app = Flask(__name__)

print("Num GPUs Available: ", len(
    tf.config.experimental.list_physical_devices('GPU')))


def load_graph_from_pb(pb_file):
    graph = tf.Graph()
    graph_def = tf.compat.v1.GraphDef()
    graph_def.ParseFromString(open(pb_file, "rb").read())

    with graph.as_default():
        tf.import_graph_def(graph_def, name="")
    return graph


def run_graph(graph, inputs_list, input_tensor_names, output_tensor_names):
    assert len(inputs_list) == len(input_tensor_names)

    input_tensors = [graph.get_tensor_by_name(
        name) for name in input_tensor_names]
    output_tensors = [graph.get_tensor_by_name(
        name) for name in output_tensor_names]
    feed_dict = dict(zip(input_tensors, inputs_list))

    with tf.compat.v1.Session(graph=graph) as sess:
        return sess.run(output_tensors, feed_dict=feed_dict)


def run_face_detector(image):
    face_detector = load_graph_from_pb(MODEL_FACE_DETECTOR)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width = rgb_image.shape[:2]

    inputs_list = [np.expand_dims(rgb_image, axis=0)]
    input_tensor_names = ["image_tensor:0"]
    output_tensor_names = ["detection_boxes:0", "detection_scores:0"]
    boxes, scores = run_graph(
        face_detector, inputs_list, input_tensor_names, output_tensor_names)

    best_rect = boxes.squeeze()[scores.squeeze().argmax()]
    best_rect = best_rect * [height, width, height, width]
    best_rect = best_rect.astype(np.int32)
    y1, x1, y2, x2 = best_rect
    return Rect.from_tl_br_points((x1, y1), (x2, y2))


def preprocess_image(image):
    face_rect = run_face_detector(image)
    w, h = face_rect.width, face_rect.height

    pad_w = int(w * 1.5)
    pad_h = int(h * 1.5)
    pad_rect = Rect(face_rect.x, face_rect.y, w + pad_w, h + pad_h)
    pad_rect.height = pad_rect.width = max(pad_rect.height, pad_rect.width)

    center_pad = Point((pad_rect.x + pad_rect.width) // 2,
                       (pad_rect.y + pad_rect.height) // 2)
    center_face = Point((face_rect.x + face_rect.width) //
                        2, (face_rect.y + face_rect.height) // 2)
    offset_center = Point(center_face.x - center_pad.x,
                          center_face.y - center_pad.y)
    pad_rect += offset_center

    im_h, im_w = image.shape[:2]
    br_x, br_y = pad_rect.br()
    left = abs(min(0, pad_rect.x))
    top = abs(min(0, pad_rect.y))
    right = max(0, br_x - im_w)
    bottom = max(0, br_y - im_h)

    padded_image = cv2.copyMakeBorder(
        image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    pad_rect.x += left
    pad_rect.y += top

    l, t = pad_rect.tl()
    r, b = pad_rect.br()

    return padded_image[t:b, l:r]


def run_icaonet(image):
    preprocessed_image = preprocess_image(image)
    resized_image = cv2.resize(
        preprocessed_image, IMAGE_SIZE, interpolation=cv2.INTER_AREA).astype(np.float32)
    input_image = np.expand_dims(
        resized_image, axis=0) / 255.0  # Normalize the image

    with tf.device('/GPU:0'):  # Force TensorFlow to use the GPU
        model = load_model(MODEL_ICAONET)
        return model.predict(input_image)


@app.route("/predict", methods=["POST"])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file:
        in_memory_file = io.BytesIO()
        file.save(in_memory_file)
        image = Image.open(in_memory_file)
        image = np.array(image)

        try:
            y_pred = run_icaonet(image)

            # Assuming your PhotographicRequirements class has the necessary attributes ordered correctly
            attribute_names = [
                "blurred",
                "looking_away",
                "ink_marked_creased",
                "unnatural_skin_tone",
                "too_dark_light",
                "washed_out",
                "pixelation",
                "hair_across_eyes",
                "eyes_closed",
                "varied_background",
                "roll_pitch_yaw",
                "flash_reflection_on_skin",
                "red_eyes",
                "shadows_behind_head",
                "shadows_across_face",
                "dark_tinted_lenses",
                "flash_reflection_on_lenses",
                "frames_too_heavy",
                "frame_covering_eyes",
                "hat_cap",
                "veil_over_face",
                "mouth_open",
                "presence_of_other_faces_or_toys"
            ]

            # Convert y_pred to a dictionary mapping each attribute name to its corresponding prediction value
            predictions = {attr: float(pred) for attr, pred in zip(
                attribute_names, y_pred.squeeze())}

            return jsonify(predictions)
        except Exception as e:
            return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
