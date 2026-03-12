# References:
#   https://github.com/HengyiWang/amb3r/blob/main/benchmark/tools/robustmvd/rmvd/viewer/__init__.py

def run_viewer(dataset, layout=None):
    import sys
    from PyQt5.QtWidgets import QApplication
    from .viewer import Viewer
    from .viewer_model import ViewerModel

    app = QApplication(sys.argv)

    viewer_model = ViewerModel(dataset=dataset, layout=layout)
    viewer = Viewer(model=viewer_model, title=dataset.full_name)
    viewer.showMaximized()

    app.exit(app.exec_())
