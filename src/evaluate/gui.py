import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
from PySide6.QtGui import QPixmap, QImage, QIcon
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from PySide6.QtCore import QObject, Signal, QRunnable, QThreadPool, Qt
from src.model.utils import haversine_distance
from src.evaluate.download_utils import *
from src.evaluate.eval_utils import *
from src.evaluate.map_utils import *
from PySide6.QtWidgets import *
from pyproj import Transformer
from configs.settings import *
import tensorflow as tf
from glob import glob
import numpy as np
import cv2
import sys


class PredictionWorker(QObject):
    finished = Signal(str, dict)


class PredictionTask(QRunnable):
    def __init__(self, mode, lat, lon, panorama_path, sliced_folder, callback):
        super().__init__()
        self.mode = mode
        self.lat = lat
        self.lon = lon
        self.panorama_path = panorama_path
        self.sliced_folder = sliced_folder
        self.callback = callback

    def run(self):
        try:
            if self.mode == "early":
                gps, softmax, heatmap = predict_panorama(MODEL_PATH("early_first"), self.panorama_path)
                km = None
                if gps is not None:
                    km = float(haversine_distance(
                        tf.constant([[self.lat, self.lon]]),
                        tf.constant([[gps[0], gps[1]]])
                    ).numpy())
                result = {"gps": gps, "softmax": softmax, "heatmap": heatmap, "km": km}

            elif self.mode == "middle":
                gps, softmax = predict_panorama_group(MODEL_PATH("middle_full_unfrozen"), self.sliced_folder)
                km = None
                if gps is not None:
                    km = float(haversine_distance(
                        tf.constant([[self.lat, self.lon]]),
                        tf.constant([[gps[0], gps[1]]])
                    ).numpy())
                result = {"gps": gps, "softmax": softmax, "heatmap": None, "km": km}
            else:
                result = {}

            self.callback(self.mode, result)
        except Exception as e:
            print(f"{self.mode} prediction failed:", e)
            self.callback(self.mode, {})


class MainWindow(QWidget):
    def __init__(self, prefill_coords=None):
        super().__init__()
        self.lon_input = None
        self.lat_input = None
        self.current_panorama = None
        self.current_coords = None
        self.map_canvas = None
        self.predicted_gps = None
        self.softmax = None
        self.predictions = {"early": None, "middle": None}
        self.predicted_heatmap = None
        self.setWindowTitle("VirgilNN")
        self.setWindowIcon(QIcon("docs/images/virgil.jpg"))
        self.setGeometry(100, 100, 1600, 900)
        self.init_ui()
        self.update_map()
        if prefill_coords:
            self.prefill_coordinates(prefill_coords)

    def init_ui(self):
        main_layout = QHBoxLayout(self)

        left_frame = QFrame()
        left_frame.setFixedWidth(int(self.width() * 0.2))
        left_layout = QVBoxLayout(left_frame)
        left_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        coord_form = QFormLayout()
        self.lat_input = QLineEdit()
        self.lat_input.setFrame(False)
        self.lon_input = QLineEdit()
        self.lon_input.setFrame(False)
        coord_form.addRow(QLabel("Latitude:"), self.lat_input)
        coord_form.addRow(QLabel("Longitude:"), self.lon_input)
        self.lat_input.textChanged.connect(self.on_coords_changed)
        self.lon_input.textChanged.connect(self.on_coords_changed)

        self.only_google_checkbox = QCheckBox()
        self.only_google_checkbox.setChecked(True)
        self.only_google_checkbox.setStyleSheet("QCheckBox { margin-right: 0; }")
        only_google_layout = QHBoxLayout()
        only_google_layout.addWidget(QLabel("Use only Google images:"))
        only_google_layout.addStretch(1)
        only_google_layout.addWidget(self.only_google_checkbox)

        self.show_grid_checkbox = QCheckBox()
        self.show_grid_checkbox.setChecked(True)
        self.show_grid_checkbox.setStyleSheet("QCheckBox { margin-right: 0; }")
        show_grid_layout = QHBoxLayout()
        show_grid_layout.addWidget(QLabel("Show grid:"))
        show_grid_layout.addStretch(1)
        show_grid_layout.addWidget(self.show_grid_checkbox)
        self.show_grid_checkbox.stateChanged.connect(self.update_map)

        self.show_heatmap_checkbox = QCheckBox()
        self.show_heatmap_checkbox.setChecked(False)
        self.show_heatmap_checkbox.setStyleSheet("QCheckBox { margin-right: 0; }")
        show_heatmap_layout = QHBoxLayout()
        show_heatmap_layout.addWidget(QLabel("Show heatmap:"))
        show_heatmap_layout.addStretch(1)
        show_heatmap_layout.addWidget(self.show_heatmap_checkbox)
        self.show_heatmap_checkbox.stateChanged.connect(lambda: self.update_panorama_display())

        self.radio_early = QRadioButton("Early Integration")
        self.radio_early.setStyleSheet("QRadioButton { spacing: 6px; }")

        self.radio_middle = QRadioButton("Middle Integration")
        self.radio_middle.setStyleSheet("QRadioButton { spacing: 6px; }")

        self.radio_early.setChecked(True)
        self.radio_group = QButtonGroup()
        self.radio_group.addButton(self.radio_early)
        self.radio_group.addButton(self.radio_middle)

        radio_layout = QHBoxLayout()
        radio_layout.addWidget(self.radio_early)
        radio_layout.addStretch(1)
        radio_layout.addWidget(self.radio_middle)

        btn_show = QPushButton("Show")
        btn_guess = QPushButton("Guess location")
        btn_show.clicked.connect(self.on_show_clicked)
        btn_guess.clicked.connect(self.on_guess_clicked)

        left_layout.addLayout(coord_form)
        left_layout.addLayout(only_google_layout)
        left_layout.addLayout(show_grid_layout)
        left_layout.addLayout(show_heatmap_layout)
        left_layout.addLayout(radio_layout)
        left_layout.addSpacing(5)
        left_layout.addWidget(btn_show)
        left_layout.addWidget(btn_guess)
        left_layout.addStretch(1)

        early_group = QGroupBox("Early Integration")
        early_form = QFormLayout()
        self.early_lat = QLineEdit()
        self.early_lat.setReadOnly(True)
        self.early_lat.setFrame(False)
        self.early_lon = QLineEdit()
        self.early_lon.setReadOnly(True)
        self.early_lon.setFrame(False)
        self.early_km = QLineEdit()
        self.early_km.setReadOnly(True)
        self.early_km.setFrame(False)
        early_form.addRow("Latitude:", self.early_lat)
        early_form.addRow("Longitude:", self.early_lon)
        early_form.addRow("Distance (km):", self.early_km)
        early_group.setLayout(early_form)

        middle_group = QGroupBox("Middle Integration")
        middle_form = QFormLayout()
        self.middle_lat = QLineEdit()
        self.middle_lat.setReadOnly(True)
        self.middle_lat.setFrame(False)
        self.middle_lon = QLineEdit()
        self.middle_lon.setReadOnly(True)
        self.middle_lon.setFrame(False)
        self.middle_km = QLineEdit()
        self.middle_km.setReadOnly(True)
        self.middle_km.setFrame(False)
        middle_form.addRow("Latitude:", self.middle_lat)
        middle_form.addRow("Longitude:", self.middle_lon)
        middle_form.addRow("Distance (km):", self.middle_km)
        middle_group.setLayout(middle_form)

        left_layout.addWidget(early_group)
        left_layout.addWidget(middle_group)

        self.tabs = QTabWidget()
        self.tabs.setTabPosition(QTabWidget.TabPosition.North)
        self.tabs.currentChanged.connect(self.on_tab_changed)
        self.tabs.setContentsMargins(0, 0, 0, 0)

        self.panorama_tab = QWidget()
        panorama_layout = QVBoxLayout(self.panorama_tab)

        self.panorama_label = QLabel()
        self.panorama_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.panorama_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self.panorama_label.setMinimumSize(1, 1)
        panorama_layout.addWidget(self.panorama_label)

        self.map_tab = QWidget()
        self.map_tab.setContentsMargins(0, 0, 0, 0)

        self.map_layout = QVBoxLayout(self.map_tab)
        self.map_layout.setContentsMargins(0, 0, 0, 0)
        self.map_layout.setSpacing(0)
        self.map_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        self.tabs.addTab(self.map_tab, "Map")
        self.tabs.addTab(self.panorama_tab, "Panorama")

        main_layout.addWidget(left_frame)
        main_layout.addWidget(self.tabs)
        self.radio_early.toggled.connect(self.update_display_from_selection)
        self.radio_middle.toggled.connect(self.update_display_from_selection)
        self.setLayout(main_layout)

    def on_coords_changed(self):
        self.predicted_heatmap = None
        self.predicted_gps = None
        self.softmax = None

        self.early_lat.clear()
        self.early_lon.clear()
        self.early_km.clear()

        self.middle_lat.clear()
        self.middle_lon.clear()
        self.middle_km.clear()

        self.update_panorama_display()
        self.update_map()

    def on_tab_changed(self, index):
        if self.tabs.widget(index) == self.panorama_tab:
            self.update_panorama_display()
        elif self.tabs.widget(index) == self.map_tab:
            self.update_map()

    def prefill_coordinates(self, coords):
        lat, lon = coords
        self.lat_input.setText(lat)
        self.lon_input.setText(lon)

    def ensure_images_for_coords(self, lat, lon):
        lat, lon = round(float(lat), 12), round(float(lon), 12)
        panorama_path = f"{GUI_IMAGES_PATH}/{lat},{lon}/panorama.jpg"
        if os.path.exists(panorama_path):
            panorama = cv2.imread(panorama_path)
            actual_lat, actual_lon = lat, lon
        else:
            actual_lat, actual_lon = download_location(lat, lon, GUI_IMAGES_PATH, self.only_google_checkbox.isChecked())
            if actual_lat is None:
                QMessageBox.warning(self, "Error", f"No nearby images found for {lat}, {lon}!")
                return None, None
            images_path = f"{GUI_IMAGES_PATH}/{actual_lat},{actual_lon}"
            panorama_path = f"{images_path}/panorama.jpg"
            sliced_path = f"{images_path}/sliced"
            images = self.load_images(sliced_path)
            if not images:
                return None, None
            panorama = self.join_images(images)
            cv2.imwrite(panorama_path, panorama)

        self.lat_input.setText(str(actual_lat))
        self.lon_input.setText(str(actual_lon))
        self.current_panorama = cv2.cvtColor(panorama, cv2.COLOR_BGR2RGB)
        self.current_coords = (actual_lat, actual_lon)
        return self.current_panorama, panorama_path

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.update_panorama_display()
        self.update_map()

    @staticmethod
    def _update_label_with_image(label, array, apply_colormap=False):
        if array is None:
            return False

        if isinstance(array, Image.Image):
            array = np.array(array)

        if apply_colormap and len(array.shape) == 2:
            array = cv2.applyColorMap(np.uint8(255 * array), cv2.COLORMAP_JET)

        if len(array.shape) == 2:
            array = cv2.cvtColor(array, cv2.COLOR_GRAY2RGB)
        elif array.shape[2] == 4:
            array = cv2.cvtColor(array, cv2.COLOR_RGBA2RGB)

        h, w = array.shape[:2]
        container_w, container_h = label.width(), label.height()

        if container_w / container_h <= w / h:
            new_w, new_h = container_w, int(container_w / w * h)
        else:
            new_w, new_h = int(container_h * w / h), container_h

        resized = cv2.resize(array, (new_w, new_h))
        img = QImage(resized.data, new_w, new_h, new_w * 3, QImage.Format.Format_RGB888)
        label.setPixmap(QPixmap.fromImage(img))
        return True

    def update_panorama_display(self, lat=None, lon=None):
        if lat is not None and lon is not None and self.current_coords != (lat, lon):
            panorama, _ = self.ensure_images_for_coords(lat, lon)
            if panorama is None:
                return False
        if self.current_panorama is None:
            return False

        img = self.current_panorama.copy()
        if self.show_heatmap_checkbox.isChecked() and self.predicted_heatmap is not None:
            heatmap_array = np.array(self.predicted_heatmap, dtype=np.float32)

            low, high = np.percentile(heatmap_array, 5), np.percentile(heatmap_array, 99)
            heatmap_array = np.clip((heatmap_array - low) / (high - low + 1e-10), 0, 1)

            heatmap_uint8 = np.uint8(255 * heatmap_array)
            heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
            heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

            if heatmap_array.shape[0] < img.shape[0] or heatmap_array.shape[1] < img.shape[1]:
                heatmap_color = cv2.GaussianBlur(heatmap_color, (3, 3), 0)

            heatmap_color = cv2.resize(heatmap_color, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC)
            heatmap_color = cv2.GaussianBlur(heatmap_color, (7, 7), 0)
            img = cv2.addWeighted(img, 0.6, heatmap_color, 0.4, 0)

        return self._update_label_with_image(self.panorama_label, img)

    def load_panorama_from_inputs(self):
        lat = self.lat_input.text().strip()
        lon = self.lon_input.text().strip()
        if not self.validate_coordinates(lat, lon):
            QMessageBox.warning(self, "Error", "Enter valid latitude and longitude values.")
            return None, None, None, None
        panorama, panorama_path = self.ensure_images_for_coords(lat, lon)
        if panorama is None:
            return None, None, None, None
        self.update_panorama_display()
        return float(lat), float(lon), panorama, panorama_path

    def on_show_clicked(self):
        lat, lon, panorama, _ = self.load_panorama_from_inputs()
        if panorama is None: return
        self.tabs.setCurrentWidget(self.panorama_tab)

    def on_guess_clicked(self):
        lat, lon, panorama, panorama_path = self.load_panorama_from_inputs()
        if panorama is None:
            return
        sliced_folder = os.path.join(os.path.dirname(panorama_path), "sliced")

        self.loading = QProgressDialog("", "Cancel", 0, 0, self)
        self.loading.setWindowModality(Qt.WindowModality.ApplicationModal)
        self.loading.setWindowTitle("Running predictions")
        self.loading.show()

        bar = self.loading.findChild(QProgressBar)
        cancel_button = self.loading.findChild(QPushButton)
        container_layout = QVBoxLayout()

        if bar:
            bar.setTextVisible(False)
            bar.setMinimumWidth(self.loading.width() - 50)
            bar.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            container_layout.addWidget(bar)

        if cancel_button:
            cancel_button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
            h_layout = QHBoxLayout()
            h_layout.addStretch()
            h_layout.addWidget(cancel_button)
            h_layout.addStretch()
            container_layout.addLayout(h_layout)

        self.loading.setLayout(container_layout)
        self.loading.setFixedSize(self.loading.size())

        self.predictions_done = {"early": False, "middle": False}
        self.predictions = {"early": None, "middle": None}

        pool = QThreadPool.globalInstance()
        pool.start(PredictionTask("early", lat, lon, panorama_path, sliced_folder, self.on_prediction_done))
        pool.start(PredictionTask("middle", lat, lon, panorama_path, sliced_folder, self.on_prediction_done))

    def on_prediction_done(self, mode, result):
        self.predictions[mode] = result
        self.predictions_done[mode] = True

        if all(self.predictions_done.values()):
            self.loading.cancel()
            self.update_display_from_selection()
            self.tabs.setCurrentWidget(self.map_tab)

    def update_display_from_selection(self):
        early_pred = self.predictions.get("early")
        middle_pred = self.predictions.get("middle")

        if early_pred and early_pred["gps"] is not None:
            lat_e, lon_e = early_pred["gps"]
            self.early_lat.setText(f"{lat_e:.6f}")
            self.early_lon.setText(f"{lon_e:.6f}")
            self.early_km.setText(f"{early_pred['km']:.2f}")
        else:
            self.early_lat.clear()
            self.early_lon.clear()
            self.early_km.clear()

        if middle_pred and middle_pred["gps"] is not None:
            lat_m, lon_m = middle_pred["gps"]
            self.middle_lat.setText(f"{lat_m:.6f}")
            self.middle_lon.setText(f"{lon_m:.6f}")
            self.middle_km.setText(f"{middle_pred['km']:.2f}")
        else:
            self.middle_lat.clear()
            self.middle_lon.clear()
            self.middle_km.clear()

        if self.radio_early.isChecked():
            self.predicted_gps = early_pred["gps"] if early_pred else None
            self.softmax = early_pred["softmax"] if early_pred else None
            self.predicted_heatmap = early_pred["heatmap"] if early_pred else None
        else:
            self.predicted_gps = middle_pred["gps"] if middle_pred else None
            self.softmax = middle_pred["softmax"] if middle_pred else None
            self.predicted_heatmap = early_pred["heatmap"] if early_pred else None

        self.update_panorama_display()
        self.update_map()

    def update_map(self):
        valid = self.validate_coordinates(self.lat_input.text(), self.lon_input.text())
        if not hasattr(self, "map_canvas") or self.map_canvas is None:
            fig, ax = plt.subplots(figsize=FIG_SIZE, facecolor="lightskyblue")
            self.map_canvas = FigureCanvas(fig)
            self.map_ax = ax
            self.map_canvas.setParent(self.map_tab)
            self.map_canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
            self.map_layout.addWidget(self.map_canvas, 1)
            self.map_canvas.mpl_connect("button_press_event", self.on_map_click)

        if valid:
            lat, lon = float(self.lat_input.text()), float(self.lon_input.text())
            draw_map(self.map_ax, lat, lon, self.show_grid_checkbox.isChecked(), self.predicted_gps, self.softmax)
        else:
            draw_map(self.map_ax, None, None, self.show_grid_checkbox.isChecked(), self.predicted_gps, self.softmax)
        self.map_canvas.draw()

    def on_map_click(self, event):
        if event.inaxes != self.map_ax or event.xdata is None or event.ydata is None:
            return
        transformer = Transformer.from_crs(PROJECTION_TYPE, "EPSG:4326", always_xy=True)
        lon, lat = transformer.transform(event.xdata, event.ydata)
        self.lat_input.setText(f"{lat:.6f}")
        self.lon_input.setText(f"{lon:.6f}")
        result = self.update_panorama_display(lat, lon)
        self.update_map()
        if result: self.tabs.setCurrentWidget(self.panorama_tab)

    @staticmethod
    def validate_coordinates(lat, lon):
        try:
            lat, lon = float(lat), float(lon)
            return -90 <= lat <= 90 and -180 <= lon <= 180
        except ValueError:
            return False

    @staticmethod
    def load_images(path):
        image_paths = sorted(glob(os.path.join(path, "*.jpg")))
        images = []
        for img_path in image_paths:
            img = cv2.imread(img_path)
            if img is not None:
                images.append(img)
        return images

    @staticmethod
    def join_images(images):
        min_h = min(img.shape[0] for img in images)
        resized = [cv2.resize(img, (int(img.shape[1] * min_h / img.shape[0]), min_h)) for img in images]
        return np.hstack(resized)


if __name__ == "__main__":
    prefill = None
    if len(sys.argv) > 1:
        try:
            lat_str, lon_str = sys.argv[1].split(",")
            prefill = (lat_str, lon_str)
        except Exception:
            print("Invalid coordinate format. Use: python main.py latitude,longitude")

    app = QApplication(sys.argv)
    app.setStyleSheet("""QLineEdit { border-radius: 4px; padding: 2px; }""")
    win = MainWindow(prefill)
    win.show()
    sys.exit(app.exec())
