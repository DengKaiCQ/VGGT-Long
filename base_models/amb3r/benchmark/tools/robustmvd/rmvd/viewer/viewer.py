#!/usr/bin/env python3

# References:
#   https://github.com/HengyiWang/amb3r/blob/main/benchmark/tools/robustmvd/rmvd/viewer/viewer.py

from PyQt5.QtWidgets import QWidget, QGridLayout
from iviz.manager import Manager
from iviz.widgets.containers import DisplayGrid

from iviz.viewers.dataset import _OversizeScrollArea
from itypes import TraceLogger
from iviz.widgets.containers import IVizArea

from .viewer_controls import ViewerControls


class Viewer(QWidget):
    def __init__(self, model, title=None, parent=None):
        self.__log = TraceLogger()
        super().__init__(parent)

        self._model = model
        self._index = None
        self._grid = None
        self._manager = None
        self._controls = None

        self._title = None
        self.set_title(title)

        self.initUI()

    def set_title(self, title):
        if title is not None:
            title = title if title.startswith("iviz") else f"iviz: {title}"
        else:
            title = "iviz"
        self._title = title
        self.setWindowTitle(self._title)

    def initUI(self):
        self._manager = Manager()

        self._init_display_grid()

        self._controls = ViewerControls(self._model)
        self._controls.index_changed.connect(self.show_index)
        self._controls.layout_changed.connect(self.update_layout)

        self._grid_scroll = _OversizeScrollArea(self._grid)

        self._area = IVizArea(self._manager)
        self._area.set_main_widget(self._grid_scroll)
        self._area.set_controls(self._controls)

        self._layout = QGridLayout()
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.addWidget(self._area)
        self.setLayout(self._layout)

        self._controls.set_index(0)

    def _init_display_grid(self):
        self._grid = DisplayGrid()
        for col, row, colspan, rowspan, display in self._model.get_displays(self._manager):
            self._grid.set_widget(widget=display, col=col, row=row, colspan=colspan, rowspan=rowspan)

    def _reset_display_grid(self):
        for display in self._grid.displays()[::-1]:
            display.setParent(None)

        self._init_display_grid()
        self._grid_scroll.setWidget(self._grid)

    def show_index(self, index, ignore_same=False):
        if self._index == index and not ignore_same: return
        self.__log.debug(f"show index {index} (old = {self._index})")

        self._index = index
        display_datas = self._model[self._index]
        # TODO: get info_dict and print it to console

        for idx, display_data in enumerate(display_datas):
            display = self._grid.displays()[idx]
            display.set_data(**display_data)

    def update_layout(self, _):
        self._reset_display_grid()
        self.show_index(self._index, ignore_same=True)
