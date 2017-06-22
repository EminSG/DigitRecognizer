import tkinter as tk


class ImageDrawer(tk.Frame):
    def __init__(self, parent):
        tk.Frame.__init__(self, parent)
        self.points = []
        self.spline = 0
        self.tag1 = "theline"

        self.c = tk.Canvas(self, bg="black", width=300, height=300)
        self.c.configure(cursor="crosshair")
        self.c.pack()
        self.c.bind("<B1-Motion>", self.pointBig)
        self.c.bind("<B3-Motion>", self.pointSmall)

    def pointBig(self, event):
        self.c.create_oval(event.x - 8, event.y - 8, event.x + 8, event.y + 8, fill="white", outline="white")
        self.points.append(event.x)
        self.points.append(event.y)

    def pointSmall(self, event):
        self.c.create_oval(event.x - 2, event.y - 2, event.x + 2, event.y + 2, fill="gray", outline="gray")

    def clear(self):
        self.points = []
        self.c.delete("all")
        self.c.update()
