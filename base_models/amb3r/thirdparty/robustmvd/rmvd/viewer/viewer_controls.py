# References:
#   https://github.com/HengyiWang/amb3r/blob/main/thirdparty/robustmvd/rmvd/viewer/viewer_controls.py

from PyQt5.QtWidgets import QWidget, QGridLayout, QPushButton, QSizePolicy, QComboBox
from PyQt5.QtCore import QTimer

from itypes import TraceLogger
from iviz.widgets.controls import IntSlider
from iviz.widgets.controls.fps_slider import FPSSlider
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtGui import QIcon
from iviz.resources import play_icon_file, previous_icon_file, next_icon_file


class ViewerControls(QWidget):
    index_changed = pyqtSignal(int)
    layout_changed = pyqtSignal(str)

    def __init__(self, model):
        self.__log = TraceLogger()
        super().__init__()

        self._model = model
        self._index = None

        self._frame_delay = 1 / 5  # 5 frames per second
        self._play_timer = QTimer()
        self._play_timer.timeout.connect(self._play_next)

        self._layout = None
        self._slider = None
        self._layout_nameDropdown = None
        self._play_button = None
        self._prev_button = None
        self._next_button = None
        self._fps = None
        self.initUI()

    def set_index(self, index):
        if self._index == index: return
        self.__log.debug(f"goto index {index} (old = {self._index})")
        self._index = index
        self._slider.change_value(index)
        self.index_changed.emit(self._index)

    def initUI(self):
        self._layout = QGridLayout()

        self._layout.setContentsMargins(0, 0, 0, 0)

        if len(self._model.get_layout_names()) > 1:
            self._layout_nameDropdown = QComboBox()
            self._layout.addWidget(self._layout_nameDropdown, 0, 0, 1, 1)
            for idx, layout in enumerate(self._model.get_layout_names()):
                self._layout_nameDropdown.addItem(layout)
                if layout == self._model.get_cur_layout_name():
                    cur_layout_idx = idx

            self._layout_nameDropdown.setCurrentIndex(cur_layout_idx)
            self._layout_nameDropdown.currentIndexChanged.connect(self.current_layout_name_dropdown_changed)

        self._slider = IntSlider((0, 0))
        self._slider.value_changed.connect(self.set_index)
        self._layout.addWidget(self._slider, 0, 2, 1, 1)
        self._layout.setColumnStretch(2, 2)
        self._slider.set_range((0, len(self._model) - 1))

        self._play_button = QPushButton("Play")
        self._play_button.setIcon(QIcon(play_icon_file.str()))
        self._play_button.setCheckable(True)
        self._play_button.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)

        def play_toggle(value):
            if value:
                self.play()
            else:
                self.stop()

        self._play_button.toggled.connect(play_toggle)
        self._layout.addWidget(self._play_button, 0, 5, 1, 1)

        self._fps = FPSSlider()
        self._fps.valueChanged.connect(self._update_fps)
        self._layout.addWidget(self._fps, 0, 6, 1, 1)

        self._prev_button = QPushButton("Prev")
        self._prev_button.setIcon(QIcon(previous_icon_file.str()))
        self._prev_button.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        self._prev_button.clicked.connect(self.previous)
        self._layout.addWidget(self._prev_button, 0, 3, 1, 1)

        self._next_button = QPushButton("Next")
        self._next_button.setIcon(QIcon(next_icon_file.str()))
        self._next_button.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        self._next_button.clicked.connect(self.next)
        self._layout.addWidget(self._next_button, 0, 4, 1, 1)

        self.setLayout(self._layout)

    def _update_fps(self, value):
        self._frame_delay = 1 / value

        if self._play_timer.isActive():
            self._play_timer.stop()
            self._play_timer.start(int(self._frame_delay * 1000))

    def next(self):
        if self._index < len(self._model) - 1:
            self.set_index(self._index + 1)

    def _play_next(self):
        if self._index < len(self._model) - 1:
            self.set_index(self._index + 1)

            if self._index == len(self._model) - 1:
                self.stop()
        else:
            self.stop()

    def play(self):
        self._play_next()
        self._play_timer.start(int(self._frame_delay * 1000))
        self._play_button.blockSignals(True)
        self._play_button.setChecked(True)
        self._play_button.blockSignals(False)

    def stop(self):
        self._play_timer.stop()
        self._play_button.blockSignals(True)
        self._play_button.setChecked(False)
        self._play_button.blockSignals(False)

    def previous(self):
        if self._index > 0:
            self.set_index(self._index - 1)

    def current_layout_name_dropdown_changed(self):
        layout_name = self._layout_nameDropdown.currentText()
        if self._model.get_cur_layout_name() == layout_name: return
        self._model.set_layout(layout_name)
        self.layout_changed.emit(layout_name)
        self.__log.debug(f"layout name changed to {layout_name}")
