from PyQt5.QtWidgets import QWidget, QLabel
from PyQt5.QtGui import QPainter, QColor, QPen, QPixmap, QImage
from PyQt5.QtCore import Qt, pyqtSignal, QPoint

import numpy as np
import cv2 as cv

class DrawingArea(QLabel):

    signal = pyqtSignal()

    def __init__(self, size):
        super().__init__()
        self.size = size
        self.setMinimumSize(size)  # Ustaw minimalny rozmiar obszaru do rysowania
        # self.drawing = False  # Flaga do śledzenia stanu rysowania
        # self.last_point = QPoint()  # Ostatni punkt, aby rysować linię

        # Lista punktów do przechowywania ścieżki rysowania
        self.points = []
        canvas = QPixmap(size)
        canvas.fill(Qt.white)
        self.setPixmap(canvas)
        self.last_x, self.last_y = None, None



    # def paintEvent(self, event):
    #     qp = QPainter(self)
    #     qp.setRenderHint(QPainter.Antialiasing)  # Włącz wygładzanie krawędzi
    #     self.drawLines(qp)

    # def drawLines(self, qp):
    #     qp.setPen(QPen(QColor(0, 0, 0), 2))  # Ustaw kolor i grubość pędzla
    #     for point in self.points:
    #         qp.drawPoint(point)  # Rysowanie punktów

    # def mousePressEvent(self, event):
    #     if event.button() == Qt.LeftButton:  # Sprawdzenie czy lewy przycisk myszy jest wciśnięty
    #         self.drawing = True
    #         self.last_point = event.pos()  # Zapamiętanie pozycji kliknięcia
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
    # def mouseMoveEvent(self, event):
    #     if self.drawing:  # Jeśli rysujemy
    #         current_point = event.pos()
    #         self.points.append(current_point)  # Dodanie bieżącego punktu do listy
    #         self.update()  # Odświeżenie widoku

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:  # Sprawdzenie czy lewy przycisk myszy został zwolniony

            self.last_x = None
            self.last_y = None

        #Update resized Image


    def reset_image(self):        
        canvas = QPixmap(self.size)
        canvas.fill(Qt.white)
        self.setPixmap(canvas)
        self.signal.emit()

    def get_image(self):

        qimage = self.pixmap().toImage()
        # Konwersja QImage do formatu RGB
        qimage = qimage.convertToFormat(QImage.Format_RGB888)
        # Pobranie danych z QImage
        width = qimage.width()
        height = qimage.height()
        ptr = qimage.bits()
        ptr.setsize(qimage.byteCount())
        # Tworzenie tablicy NumPy
        img = np.array(ptr).reshape(height, width, 3)  # Zmiana kształtu na (wysokość, szerokość, kanały)
        img = cv.cvtColor(img, cv.COLOR_RGB2BGR)  # Konwersja z RGB do BGR (OpenCV używa BGR)
        return img

    def get_pixmap(self) -> QPixmap:
        return self.pixmap()
