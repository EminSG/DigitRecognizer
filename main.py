import tkinter as tk
import NeuralNetworkMnist as nn
import TinkerTable as table
import ImageDrawer as drawer
import numpy
from PIL import Image, ImageGrab


class App:
    def __init__(self):
        self.nn = nn.NeuralNetworkMnist(self)
        self.rootFrame = tk.Frame(tk.Tk(), width=500, padx=10, pady=10)
        self.rootFrame.grid_rowconfigure(5)
        self.rootFrame.grid_columnconfigure(2)
        # images that trained
        self.imagesGrid = table.SimpleTable(self.rootFrame, rows=7, columns=7)
        self.imagesGrid.grid(row=1, column=1, padx=10, pady=10)
        # where to draw image
        inputNumberLabel = tk.Label(self.rootFrame, text="Draw number below:")
        inputNumberLabel.grid(row=0, column=0)

        self.imageDrawer = drawer.ImageDrawer(self.rootFrame)
        self.imageDrawer.grid(row=1, column=0, padx=10, pady=10)
        # train button
        train = tk.Button(self.rootFrame, text="Train", command=self.nn.train, width=50)
        train.grid(row=0, column=1)
        # show train step
        self.interationsLabel = tk.Label(self.rootFrame, text="")
        self.interationsLabel.grid(row=2, column=2)

        clearInput = tk.Button(self.rootFrame, text="Clear", command=self.imageDrawer.clear, width=10)
        clearInput.grid(row=2, column=0)

        showResult = tk.Button(self.rootFrame, text="Result", command=self.showResultForImage, width=10)
        showResult.grid(row=2, column=1)

        self.resultNumberLabel = tk.Label(self.rootFrame, text="")
        self.resultNumberLabel.grid(row=4, column=0)

        self.rootFrame.pack()
        self.imagesGrid.mainloop()


    @staticmethod
    def createImage(pixels):
        im = Image.new('1', (28, 28), color=255)
        im.putdata(pixels, scale=255)
        return im

    def updateInputImages(self, images):
        for i in range(7):
            for j in range(7):
                pilImage = self.createImage(pixels=images[i*7 + j])
                self.imagesGrid.set(i, j, pilImage)

    def updateIteration(self, i):
        self.interationsLabel.configure(text="Iteration number " + str(i))

    def showResultForImage(self):
        image = self.getImageFromCanvas()
        self.nn.resultForImage(image)
        self.resultNumberLabel.update()

    def getImageFromCanvas(self):
        x = self.rootFrame.winfo_rootx() + self.imageDrawer.winfo_x()
        y = self.rootFrame.winfo_rooty() + self.imageDrawer.winfo_y()
        x1 = x + self.imageDrawer.winfo_width()
        y1 = y + self.imageDrawer.winfo_height()

        image = ImageGrab.grab().crop((x, y, x1, y1)).resize((28, 28)).convert('L')
        pixels = numpy.asarray(image) / 255
        pixels = numpy.reshape(pixels, (-1, 28 * 28))

        return pixels.astype(dtype=numpy.float32)

App()

