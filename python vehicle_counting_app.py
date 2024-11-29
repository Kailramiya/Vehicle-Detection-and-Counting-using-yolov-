import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk


class CentroidTracker:
    def __init__(self, maxDisappeared=50):
        self.nextObjectID = 0
        self.objects = {}
        self.disappeared = {}
        self.maxDisappeared = maxDisappeared

    def register(self, centroid):
        self.objects[self.nextObjectID] = centroid
        self.disappeared[self.nextObjectID] = 0
        self.nextObjectID += 1

    def deregister(self, objectID):
        del self.objects[objectID]
        del self.disappeared[objectID]

    def update(self, rects):
        if len(rects) == 0:
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)
            return self.objects

        inputCentroids = np.zeros((len(rects), 2), dtype="int")
        for (i, (xA, yA, xB, yB)) in enumerate(rects):
            cX = int((xA + xB) / 2.0)
            cY = int((yA + yB) / 2.0)
            inputCentroids[i] = (cX, cY)

        if len(self.objects) == 0:
            for i in range(0, len(inputCentroids)):
                self.register(inputCentroids[i])
        else:
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())

            D = np.zeros((len(objectCentroids), len(inputCentroids)), dtype="float")

            for i in range(len(objectCentroids)):
                for j in range(len(inputCentroids)):
                    D[i, j] = np.linalg.norm(np.array(objectCentroids[i]) - np.array(inputCentroids[j]))

            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            usedRows = set()
            usedCols = set()

            for (row, col) in zip(rows, cols):
                if row in usedRows or col in usedCols:
                    continue

                objectID = objectIDs[row]
                self.objects[objectID] = inputCentroids[col]
                self.disappeared[objectID] = 0

                usedRows.add(row)
                usedCols.add(col)

            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            for row in unusedRows:
                objectID = objectIDs[row]
                self.disappeared[objectID] += 1
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)

            unusedCols = set(range(0, D.shape[1])).difference(usedCols)
            for col in unusedCols:
                self.register(inputCentroids[col])

        return self.objects


class VehicleCountingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Vehicle Detection and Counting")
        self.root.geometry("800x600")

        self.video_path = ""
        self.cap = None
        self.running = False

        # Initialize YOLOv3 model
        self.net = cv2.dnn.readNet(r"C:\Users\amank\Downloads\new\Vehicle-Detection-and-Counting-using-YOLO-V3-main\yolo-coco\yolov3.weights", r"C:\Users\amank\Downloads\new\Vehicle-Detection-and-Counting-using-YOLO-V3-main\yolo-coco\yolov3.cfg")
        self.output_layers = self.net.getUnconnectedOutLayersNames()

        self.vehicle_classes = ["car", "motorbike", "bus", "truck"]
        with open(r"C:\Users\amank\Downloads\new\Vehicle-Detection-and-Counting-using-YOLO-V3-main\yolo-coco\coco.names", "r") as f:
            self.classes = [line.strip() for line in f.readlines()]

        self.ct = CentroidTracker(maxDisappeared=50)

        # Video display area
        self.video_label = tk.Label(self.root)
        self.video_label.pack(side=tk.LEFT)

        # Vehicle counts display
        self.count_display = tk.Label(self.root, text="", font=("Helvetica", 16))
        self.count_display.pack(side=tk.LEFT, padx=10)

        # Trackbar for video length
        self.trackbar = tk.Scale(self.root, from_=0, to=100, orient=tk.HORIZONTAL, length=400, sliderlength=20)
        self.trackbar.pack(pady=20)

        # Current time and total time labels
        self.current_time_label = tk.Label(self.root, text="Current Time: 00:00:00", font=("Helvetica", 10))
        self.current_time_label.pack(side=tk.LEFT, padx=(10, 0))

        self.total_time_label = tk.Label(self.root, text="Total Time: 00:00:00", font=("Helvetica", 10))
        self.total_time_label.pack(side=tk.RIGHT, padx=(0, 10))

        # Browse and Stop buttons
        self.browse_button = tk.Button(self.root, text="Browse", command=self.browse_video)
        self.browse_button.pack(pady=10)

        self.stop_button = tk.Button(self.root, text="Stop", command=self.stop_video)
        self.stop_button.pack(pady=10)

    def browse_video(self):
        self.video_path = filedialog.askopenfilename()
        if self.video_path:
            self.start_video()

    def start_video(self):
        self.cap = cv2.VideoCapture(self.video_path)
        self.running = True
        self.counted_ids = set()
        self.vehicle_count = {"car": 0, "motorbike": 0, "bus": 0, "truck": 0}
        self.line_position = 200  # Y-coordinate of the counting line
        self.play_video()

# Update the play_video method in your VehicleCountingApp class
    def play_video(self):
        if not self.running:
            return

        ret, frame = self.cap.read()
        if not ret:
            messagebox.showinfo("End", "Video has ended.")
            self.stop_video()
            return

        frame = cv2.resize(frame, (640, 480))
        boxes, class_ids = self.detect_vehicles(frame)

        # Draw bounding boxes for all detected vehicles
        for i, box in enumerate(boxes):
            x, y, w, h = box
            class_id = class_ids[i]
            class_name = self.classes[class_id]

            if class_name in self.vehicle_classes:  # Check if the detected object is a vehicle
                # Draw bounding box for detected vehicle
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, class_name, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Track vehicles using CentroidTracker
        rects = [(x, y, x + w, y + h) for (x, y, w, h) in boxes]
        objects = self.ct.update(rects)

        # Track and count vehicles that cross the line
        for objectID, centroid in objects.items():
            if centroid[1] > self.line_position and objectID not in self.counted_ids:
                # Check if the detected object is a vehicle and count it
                if objectID in objects:
                    self.counted_ids.add(objectID)

                    # Get the class_id corresponding to the objectID
                    for i, box in enumerate(boxes):
                        x, y, w, h = box
                        if x < centroid[0] < x + w and y < centroid[1] < y + h:
                            class_name = self.classes[class_ids[i]]
                            if class_name in self.vehicle_classes:
                                self.vehicle_count[class_name] += 1
                                break

        # Draw the line on the frame
        cv2.line(frame, (0, self.line_position), (frame.shape[1], self.line_position), (0, 0, 255), 2)

        # Update the trackbar and vehicle count display
        current_frame = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
        total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        total_time_seconds = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) / self.cap.get(cv2.CAP_PROP_FPS))
        current_time_seconds = int(current_frame / self.cap.get(cv2.CAP_PROP_FPS))

        self.trackbar.config(to=total_frames)
        self.trackbar.set(current_frame)
        self.count_display.config(text=self.get_vehicle_counts())
        self.update_time_labels(current_time_seconds, total_time_seconds)

        # Convert frame to PhotoImage and display
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        img_tk = ImageTk.PhotoImage(image=img)
        self.video_label.imgtk = img_tk
        self.video_label.configure(image=img_tk)

        self.root.after(1, self.play_video)



    def stop_video(self):
        self.running = False
        if self.cap is not None:
            self.cap.release()
        self.video_label.configure(image='')

    def get_vehicle_counts(self):
        counts = "\n".join([f"{vehicle}: {self.vehicle_count[vehicle]}" for vehicle in self.vehicle_classes])
        total_count = sum(self.vehicle_count.values())
        counts += f"\nTotal Vehicles: {total_count}"
        return counts

    def update_time_labels(self, current_time_seconds, total_time_seconds):
        current_time_str = self.seconds_to_time_format(current_time_seconds)
        total_time_str = self.seconds_to_time_format(total_time_seconds)

        self.current_time_label.config(text=f"Current Time: {current_time_str}")
        self.total_time_label.config(text=f"Total Time: {total_time_str}")

    def seconds_to_time_format(self, seconds):
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        seconds = seconds % 60
        return f"{hours:02}:{minutes:02}:{seconds:02}"

    def detect_vehicles(self, frame):
        height, width = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        self.net.setInput(blob)
        outs = self.net.forward(self.output_layers)

        class_ids = []
        confidences = []
        boxes = []

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5 and self.classes[class_id] in self.vehicle_classes:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        # Use the NMS function and handle its output correctly
        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        # Check if indices is not empty and flatten correctly
        if len(indices) > 0:
            indices = indices.flatten()
            boxes = [boxes[i] for i in indices]
            class_ids = [class_ids[i] for i in indices]

        return boxes, class_ids


if __name__ == "__main__":
    root = tk.Tk()
    app = VehicleCountingApp(root)
    root.mainloop()
