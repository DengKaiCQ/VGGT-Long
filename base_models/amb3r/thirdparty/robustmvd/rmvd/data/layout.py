# References:
#   https://github.com/HengyiWang/amb3r/blob/main/thirdparty/robustmvd/rmvd/data/layout.py

import dill as pickle


def create_display_by_visualization_type(visualization_type, manager, label):
    visualization_type = visualization_type.lower()

    if visualization_type == 'flow':
        from iviz.widgets.displays import FlowDisplay
        from iviz.renderers import FlowPixmapVisualization
        pixviz = FlowPixmapVisualization()
        return FlowDisplay(manager, pixviz, label=label)

    elif visualization_type == 'img' or visualization_type == 'image':
        from iviz.widgets.displays import ImageDisplay
        from iviz.renderers import ImagePixmapVisualization
        pixviz = ImagePixmapVisualization()
        return ImageDisplay(manager, pixviz, label=label)

    else:
        from iviz.widgets.displays import FloatDisplay
        from iviz.renderers import FloatPixmapVisualization
        pixviz = FloatPixmapVisualization()
        return FloatDisplay(manager, pixviz, label=label)


class Visualization:
    def __init__(self, col, row, visualization_type, load_fct, name=None, colspan=1, rowspan=1):
        self.col = col
        self.row = row
        self.visualization_type = visualization_type
        self.load_fct = load_fct
        self.name = name
        self.colspan = colspan
        self.rowspan = rowspan

    def create_display(self, manager):
        display = create_display_by_visualization_type(self.visualization_type, manager, label=self.name)
        return display


class Layout:
    def __init__(self, name, visualizations=None):
        self.name = name
        self.visualizations = [] if visualizations is None else visualizations

    def load(self, data):
        display_datas = []

        for visualization in self.visualizations:
            display_data = visualization.load_fct(data)
            display_datas.append(display_data)

        return display_datas

    def write(self, path):
        path = path + '.pickle' if not path.endswith('.pickle') else path
        self.visualizations.append(self.name)  # TODO: refactor this

        with open(path, 'wb') as file:
            pickle.dump(self.visualizations, file)

        self.visualizations = self.visualizations[:-1]

    @classmethod
    def from_file(cls, path, name=None):
        path = path + '.pickle' if not path.endswith('.pickle') else path

        with open(path, 'rb') as file:
            visualizations = pickle.load(file)

        name = name if name is not None else visualizations[-1]
        visualizations = visualizations[:-1]

        return cls(name=name, visualizations=visualizations)
