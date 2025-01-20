from application.UI_MainWindow import Ui_MainWindow
from PyQt5 import QtWidgets
import argparse
import sys


def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path_to_model",
        type=str,
        default="models/model_with_classes.pth",
        help="The path to the model to load.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    arg = arg_parse()
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow(path_to_model=arg.path_to_model)
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
