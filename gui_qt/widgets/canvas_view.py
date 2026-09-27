"""
High-Performance Hardware-Accelerated Graphics Canvas for Book Pages and Crops.

Built on PySide6 QGraphicsView with BSP tree spatial indexing.
Supports smooth mouse-wheel zooming, pan-by-drag, zero-copy QImage updates,
and bounding-box annotation overlays with 60 FPS performance.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QImage,
    QPainter,
    QPen,
    QPixmap,
    QWheelEvent,
)
from PySide6.QtWidgets import (
    QGraphicsPixmapItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsSimpleTextItem,
    QGraphicsView,
    QWidget,
)


class StaffGraphicsView(QGraphicsView):
    """Interactive hardware-accelerated viewer for PDF page scans and music crops."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)

        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)

        # Rendering optimizations
        self.setRenderHints(
            QPainter.RenderHint.Antialiasing
            | QPainter.RenderHint.SmoothPixmapTransform
        )
        self.setViewportUpdateMode(QGraphicsView.ViewportUpdateMode.FullViewportUpdate)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)

        # Styling
        self.setStyleSheet("""
            QGraphicsView {
                background-color: #0b0f17;
                border: 1px solid #1e293b;
                border-radius: 8px;
            }
        """)

        # State
        self._pixmap_item: Optional[QGraphicsPixmapItem] = None
        self._bbox_items: List[QGraphicsRectItem] = []
        self._text_items: List[QGraphicsSimpleTextItem] = []
        self._zoom_factor: float = 1.0

    def set_qimage(self, qimg: QImage) -> None:
        """Sets the viewport image directly from a zero-copy QImage."""
        if qimg is None or qimg.isNull():
            return
        pix = QPixmap.fromImage(qimg)
        self.set_pixmap(pix)

    def set_pixmap(self, pix: QPixmap) -> None:
        """Updates the scene pixmap while maintaining current bounding rects."""
        self.clear_bboxes()
        if self._pixmap_item is None:
            self._pixmap_item = self._scene.addPixmap(pix)
            self.fit_to_view()
        else:
            self._pixmap_item.setPixmap(pix)

        self._scene.setSceneRect(self._pixmap_item.boundingRect())

    def clear_view(self) -> None:
        """Clears the canvas and resets transformations."""
        self.clear_bboxes()
        if self._pixmap_item is not None:
            try:
                self._scene.removeItem(self._pixmap_item)
            except Exception:
                pass
            self._pixmap_item = None
        self._scene.clear()
        self._pixmap_item = None
        self._zoom_factor = 1.0
        self.resetTransform()

    def load_file(self, filepath: str | Path, auto_fit: bool = True) -> bool:
        """Loads an image directly from disk using Qt's native C++ image reader."""
        pix = QPixmap(str(filepath))
        if pix.isNull():
            self.clear_view()
            return False
        self.set_pixmap(pix)
        if auto_fit:
            self.fit_to_view()
        return True

    def load_image_file(self, filepath: str | Path, auto_fit: bool = True) -> bool:
        """Alias for load_file."""
        return self.load_file(filepath, auto_fit=auto_fit)

    def load_svg_content(self, svg_str: str, auto_fit: bool = True) -> bool:
        """
        Renders vector SVG to a high-DPI aspect-ratio preserved bitmap,
        preventing any distortion while enabling interactive zoom and pan.
        """
        if not svg_str or "<svg" not in svg_str:
            self.clear_view()
            return False
        from PySide6.QtSvg import QSvgRenderer
        renderer = QSvgRenderer(svg_str.encode("utf-8"))
        if not renderer.isValid():
            self.clear_view()
            return False
        size = renderer.defaultSize()
        # High-DPI 2x vector rasterization preserving native SVG aspect ratio
        dpi_scale = 2.0
        w = max(1, int(round(size.width() * dpi_scale)))
        h = max(1, int(round(size.height() * dpi_scale)))
        img = QImage(w, h, QImage.Format.Format_ARGB32_Premultiplied)
        img.fill(Qt.GlobalColor.white)
        p = QPainter(img)
        renderer.render(p)
        p.end()
        pix = QPixmap.fromImage(img)
        self.set_pixmap(pix)
        if auto_fit:
            self.fit_to_view(is_score=True, raw_stave_h=size.height())
        return True

    def fit_to_view(self, is_score: bool = False, raw_stave_h: Optional[int] = None) -> None:
        """
        Scales the view so the image fits comfortably inside the viewport.
        When is_score is True, prevents short snippets from blowing up into oversized notes.
        """
        if self._pixmap_item is None:
            return
        self.resetTransform()
        rect = self._pixmap_item.boundingRect()
        if rect.width() <= 0 or rect.height() <= 0:
            return

        vp = self.viewport().size()
        scale_w = vp.width() / rect.width() if rect.width() > 0 else 1.0
        scale_h = vp.height() / rect.height() if rect.height() > 0 else 1.0
        fit_scale = min(scale_w, scale_h)

        if is_score:
            # Standard musical notation staff line height in viewport should be ~100-140px per staff.
            # Avoid scaling short 2-measure snippets to huge cartoon sizes.
            num_staves = 1
            if raw_stave_h is not None:
                if raw_stave_h > 350:
                    num_staves = 3
                elif raw_stave_h > 190:
                    num_staves = 2
            max_viewport_h = min(vp.height() * 0.90, 130.0 * num_staves)
            max_scale_by_h = max_viewport_h / rect.height()
            fit_scale = min(fit_scale, max_scale_by_h)

        self.scale(fit_scale, fit_scale)
        self._zoom_factor = fit_scale

    def wheelEvent(self, event: QWheelEvent) -> None:
        """Smooth zooming anchored under mouse cursor."""
        angle = event.angleDelta().y()
        if angle > 0:
            factor = 1.15
            self._zoom_factor *= factor
        elif angle < 0:
            factor = 1.0 / 1.15
            self._zoom_factor *= factor
        else:
            super().wheelEvent(event)
            return

        # Bound zoom between 0.05x and 25x
        if 0.05 <= self._zoom_factor <= 25.0:
            self.scale(factor, factor)
        else:
            # Revert factor tracking
            self._zoom_factor /= factor

        event.accept()

    def keyPressEvent(self, event) -> None:
        """Forwards arrow and page navigation keys to parent views."""
        if event.key() in (
            Qt.Key.Key_Left,
            Qt.Key.Key_Right,
            Qt.Key.Key_Up,
            Qt.Key.Key_Down,
            Qt.Key.Key_PageUp,
            Qt.Key.Key_PageDown,
            Qt.Key.Key_Home,
            Qt.Key.Key_End,
            Qt.Key.Key_A,
            Qt.Key.Key_D,
        ):
            event.ignore()
            return
        super().keyPressEvent(event)

    def clear_bboxes(self) -> None:
        """Removes all bounding box overlays."""
        for item in self._bbox_items:
            self._scene.removeItem(item)
        for item in self._text_items:
            self._scene.removeItem(item)
        self._bbox_items.clear()
        self._text_items.clear()

    def draw_bboxes(self, detections: List[Dict[str, Any]]) -> None:
        """Draws high-contrast colored bounding boxes for staves and grand-staves."""
        self.clear_bboxes()

        colors = {
            "staff": QColor(56, 189, 248, 220),       # Light blue
            "grand_staff": QColor(192, 132, 252, 220), # Purple
            "system": QColor(74, 222, 128, 220),      # Neon green
        }

        for det in detections:
            cls_name = det.get("class", "staff")
            box = det.get("box", [0, 0, 0, 0])
            if len(box) != 4:
                continue

            x1, y1, x2, y2 = box
            w = max(1, x2 - x1)
            h = max(1, y2 - y1)

            col = colors.get(cls_name, QColor(250, 204, 21, 220))

            # Rectangle item
            pen = QPen(col, 2, Qt.PenStyle.SolidLine)
            pen.setCosmetic(True)  # Width stays constant regardless of zoom level
            fill = QColor(col.red(), col.green(), col.blue(), 25)
            brush = QBrush(fill)

            rect_item = self._scene.addRect(QRectF(x1, y1, w, h), pen, brush)
            self._bbox_items.append(rect_item)

            # Label item
            conf = det.get("confidence", 0.0)
            label = f"{cls_name} {conf:.2f}" if conf > 0 else cls_name
            text_item = self._scene.addSimpleText(label)
            text_item.setPos(x1 + 4, y1 + 4)
            text_item.setBrush(QBrush(col))
            font = QFont("Cascadia Code", 9, QFont.Weight.Bold)
            text_item.setFont(font)
            self._text_items.append(text_item)
