from PyQt5.QtWidgets import QWidget, QLabel
from PyQt5.QtGui import QPainter, QColor, QPen, QPixmap, QImage
from PyQt5.QtCore import Qt, pyqtSignal, QPoint

import numpy as np
import cv2 as cv

"""This is a class which inherits from QLabel and is used to draw on the screen."""
class DrawingArea(QLabel):

    signal = pyqtSignal()

    def __init__(self, size):
        super().__init__()
        self.size = size
        self.setMinimumSize(size)
        self.points = []
        canvas = QPixmap(size)
        canvas.fill(Qt.white)
        self.setPixmap(canvas)
        self.last_x, self.last_y = None, None

    def mouseMoveEvent(self, event):

        # if event.button() != Qt.LeftButton:  # Sprawdzenie czy lewy przycisk myszy został zwolniony
        #     return
        if self.last_x is None: # First event.
            self.last_x = event.x()
            self.last_y = event.y()
            return # Ignore the first time.

        painter = QPainter(self.pixmap())
        painter.setPen(QPen(QColor(0,0,0), 3))
        painter.drawLine(self.last_x, self.last_y, event.x(), event.y())
        painter.end()
        self.update()

        # Update the origin for next time.
        self.last_x = event.x()
        self.last_y = event.y()

        self.signal.emit()
    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:  # Sprawdzenie czy lewy przycisk myszy został zwolniony

            self.last_x = None
            self.last_y = None

    """Set the image to on the drawing area to white."""
    def reset_image(self):        
        canvas = QPixmap(self.size)
        canvas.fill(Qt.white)
        self.setPixmap(canvas)
        self.signal.emit()

    """Return the image as a numpy array."""
    def get_image(self):

        qimage = self.pixmap().toImage()
        qimage = qimage.convertToFormat(QImage.Format_RGB888)
        width = qimage.width()
        height = qimage.height()
        ptr = qimage.bits()
        ptr.setsize(qimage.byteCount())
        img = np.array(ptr).reshape(height, width, 3) 
        img = cv.cvtColor(img, cv.COLOR_RGB2BGR) 
        return img

    """Return the image as a QPixmap."""
    def get_pixmap(self) -> QPixmap:
        return self.pixmap()
