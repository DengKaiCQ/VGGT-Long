# References:
#   https://github.com/HengyiWang/amb3r/blob/main/thirdparty/robustmvd/rmvd/viewer/viewer_model.py

class ViewerModel:
    def __init__(self, dataset, layout=None):
        self._dataset = dataset
        self._layout = None
        self.set_layout(layout)

    def get_layout_names(self):
        return self._dataset.get_layout_names()

    def get_cur_layout_name(self):
        return self._layout.name

    def set_layout(self, layout):
        if layout is None or isinstance(layout, str):
            self._layout = self._dataset.get_layout(layout)
        else:
            self._layout = layout

    def get_displays(self, manager):
        displays = []
        for visualization in self._layout.visualizations:
            display = visualization.create_display(manager)
            displays.append((visualization.col, visualization.row, visualization.colspan, visualization.rowspan, display))
        return displays

    @property
    def num_items(self):
        """Returns the number of items in the dataset."""
        return len(self._dataset)

    def __len__(self):
        return self.num_items

    def __getitem__(self, index):
        """Returns the display data for the given index."""
        data = self._dataset[index]
        display_data = self._layout.load(data)
        return display_data
