import tkinter as tk
from PIL import ImageTk, Image

listImages = []
class SimpleTable(tk.Frame):
    def __init__(self, parent, rows=10, columns=10):
        # use black background so it "peeks through" to
        # form grid lines
        self.rows = rows
        self.columns = columns
        tk.Frame.__init__(self, parent, background="black")
        self._widgets = []
        for row in range(rows):
            current_row = []
            for column in range(columns):
                label = tk.Label(self)
                label.grid(row=row, column=column, sticky="nsew")
                current_row.append(label)
            self._widgets.append(current_row)

        for column in range(columns):
            self.grid_columnconfigure(column)

    def set(self, row, column, new_image):
        if len(listImages) < self.rows * self.columns:
            listImages.append(new_image)
            return
        else:
            listImages.pop(row*self.columns+column)
            listImages.insert(row*self.columns+column, ImageTk.BitmapImage(new_image))
        widget = self._widgets[row][column]
        widget["image"] = listImages[row * self.columns + column]
        widget.update()
