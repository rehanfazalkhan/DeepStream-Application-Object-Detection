import sys
import gi

gi.require_version('Gst', '1.0')
gi.require_version('GstApp', '1.0')
from gi.repository import Gst, GstApp, GLib
import torch
import cv2
from collections import defaultdict
import ssl

ssl._create_default_https_context = ssl._create_unverified_context


class VideoObjectTracker:
    def __init__(self, model_name='yolov5s', class_labels=None):
        self.model = torch.hub.load('ultralytics/yolov5', model_name)
        if class_labels is None:
            self.class_labels = ['person', 'car', 'truck', 'motorcycle', 'bus']
        else:
            self.class_labels = class_labels
        self.class_dict = {int(k): v for k, v in self.model.names.items()}
        self.colors = {
            'person': (0, 255, 0),  # Green
            'car': (255, 0, 0),  # Blue
            'truck': (0, 0, 255),  # Red
            'motorcycle': (0, 255, 255),  # Yellow
            'bus': (255, 255, 0)  # Cyan
        }
        self.object_ids = {}
        self.out = None

    def load_model(self):
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

    def update_tracker(self, detected_objects, confidences, bounding_boxes):
        object_tracker = defaultdict(list)
        object_count = {label: 0 for label in self.class_labels}

        for detected_object, confidence, bbox in zip(detected_objects, confidences, bounding_boxes):
            object_id_parts = detected_object.split('_')
            if len(object_id_parts) == 1:
                object_id = detected_object + '_0.0'
                confidence = 0.0
            else:
                object_id, confidence = detected_object.split('_')
            object_id = object_id.split(':')[0]  # Remove confidence value from label

            if object_id not in self.object_ids:
                self.object_ids[object_id] = str(uuid.uuid4().hex)[:8]

            if object_id_parts[0] in self.class_labels:
                if object_id_parts[0] not in object_count:
                    object_count[object_id_parts[0]] = 1
                else:
                    object_count[object_id_parts[0]] += 1

                object_tracker[object_id_parts[0]].append({
                    'id': self.object_ids[object_id],
                    'class': object_id_parts[0],
                    'confidence': float(confidence),
                    'bbox': bbox
                })

        return object_tracker, object_count

    def draw_bounding_boxes(self, frame, object_tracker, object_count):
        for object_id, objects in object_tracker.items():
            for info in objects:
                object_label = info['class']
                confidence = info['confidence']

                x1, y1, x2, y2 = map(int, info['bbox'])
                color = self.colors.get(object_label, (0, 0, 0))  # Use default color if class not in self.colors
                thickness = 2
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

                object_id_label = f'{object_label.capitalize()} ID: {info["id"]}'
                label = f'{object_id_label} {confidence:.2f} Count: {object_count[object_label]}'
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness)

    def process_video(self, video_path, output_path='output_video.mp4'):
        vid = cv2.VideoCapture(video_path)
        if not vid.isOpened():
            raise IOError("Couldn't open video")

        frames_per_second = int(vid.get(cv2.CAP_PROP_FPS))
        frame_width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), frames_per_second,
                                   (frame_width, frame_height))

        object_tracker = defaultdict(list)
        object_count = {label: 0 for label in self.class_labels}

        while True:
            ret, frame = vid.read()
            if not ret:
                break

            results = self.model(frame)
            detected_objects = [self.class_dict[class_id] for class_id in results.pred[0][:, -1].tolist()]
            confidences = results.pred[0][:, -1].tolist()
            detected_objects_with_confidence = [
                (detected_object, confidence, bbox) for detected_object, confidence, bbox in
                zip(detected_objects, confidences, results.pred[0][:, :4].tolist())
                if detected_object in self.class_labels
            ]

            object_tracker.clear()
            object_count.clear()

            for detected_object, confidence, bbox in detected_objects_with_confidence:
                object_id_parts = detected_object.split('_')
                if len(object_id_parts) == 1:
                    object_id = detected_object + '_0.0'
                    confidence = 0.0
                else:
                    object_id, confidence = detected_object.split('_')
                object_id = object_id.split(':')[0]  # Remove confidence value from label

                if object_id not in self.object_ids:
                    self.object_ids[object_id] = str(uuid.uuid4().hex)[:8]

                if object_id_parts[0] in self.class_labels:
                    if object_id_parts[0] not in object_count:
                        object_count[object_id_parts[0]] = 1
                    else:
                        object_count[object_id_parts[0]] += 1

                    object_tracker[object_id_parts[0]].append({
                        'id': self.object_ids[object_id],
                        'class': object_id_parts[0],
                        'confidence': float(confidence),
                        'bbox': bbox
                    })

            self.draw_bounding_boxes(frame, object_tracker, object_count)

            # Display real-time count of each object type
            count_label = ', '.join([f'{label}: {count}' for label, count in object_count.items()])
            cv2.putText(frame, count_label, (10, frame_height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            self.out.write(frame)

        vid.release()
        self.out.release()


class DeepStreamApp:
    def __init__(self, model_path, class_labels, output_path='output_video.mp4'):
        self.model_path = model_path
        self.class_labels = class_labels
        self.output_path = output_path
        self.pipeline = None
        self.tracker = VideoObjectTracker(class_labels=class_labels)
        self.tracker.load_model()

        Gst.init(None)

    def _handle_buffer(self, src, buffer):
        try:
            batch_meta = GstVideo.buffer_get_video_meta(buffer)
            frame_number = batch_meta.buffer_id

            # Convert GStreamer buffer to OpenCV format
            caps_format = batch_meta.get_caps().get_structure(0)
            width = caps_format.get_int('width')
            height = caps_format.get_int('height')
            numpy_array = GstVideo.video_frame_map(buffer, Gst.MapFlags.READ)
            frame = numpy_array.data.reshape(height, width, -1)

            # Process the frame using the VideoObjectTracker
            object_tracker, object_count = self.tracker.update_tracker(
                detected_objects,
                confidences,
                bounding_boxes
            )
            self.tracker.draw_bounding_boxes(frame, object_tracker, object_count)

            # Convert OpenCV frame back to GStreamer buffer
            success, buffer = GstVideo.video_frame_map_idem(buffer, numpy_array)
            if not success:
                raise Exception("Failed to map GStreamer buffer to OpenCV format")

        except Exception as e:
            print("Error handling buffer:", str(e))

        return Gst.FlowReturn.OK

    def run(self, video_path):
        self.pipeline = Gst.parse_launch(
            f'filesrc location={video_path} ! qtdemux ! h264parse ! nvv4l2decoder ! '
            f'nvvideoconvert ! video/x-raw, format=BGRx ! appsink name=appsink sync=false'
        )

        appsink = self.pipeline.get_by_name('appsink')
        appsink.set_property('emit-signals', True)
        appsink.set_property('max-buffers', 1)
        appsink.set_property('drop', True)

        appsink.connect('new-sample', self._handle_buffer)

        self.pipeline.set_state(Gst.State.PLAYING)

        try:
            loop = GLib.MainLoop()
            loop.run()
        except KeyboardInterrupt:
            pass
        finally:
            self.pipeline.set_state(Gst.State.NULL)



if __name__ == "__main__":
    # Read configuration from the DeepStream configuration file
    with open("deepstream_app_config.txt", "r") as config_file:
        config = config_file.read()

    # Assuming the configuration file follows the format mentioned in the question
    config_dict = {}
    for line in config.splitlines():
        key, value = line.split("=")
        config_dict[key.strip()] = value.strip()

    video_path = config_dict.get("video_path", "/path/to/your/video.mp4")
    output_path = config_dict.get("output_path", "output_video.mp4")

    class_labels = ['person', 'car', 'truck', 'motorcycle', 'bus']
    object_tracker = VideoObjectTracker(class_labels=class_labels)
    object_tracker.process_video(video_path, output_path)