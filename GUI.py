from tkinter import *
import numpy as np
from PIL import ImageGrab
from Prediction import predict

window = Tk()
window.title("Prescription recognition")
widthOfEachWord = 100


l1 = Label()
a = 1


def MyProject():
    global image_number, l1, a
    widget = cv
    result = ""

    for i in range(0, 10):
        x = window.winfo_rootx() + widget.winfo_x() + i * widthOfEachWord + 2
        y = window.winfo_rooty() + widget.winfo_y() + 2
        x1 = (x + widget.winfo_width()) - (9 - i) * widthOfEachWord
        y1 = (y + widget.winfo_height()) - 4
        if i >= 1:
            x1 -= i * widthOfEachWord

        img = ImageGrab.grab().crop((x, y, x1 - 8, y1)).resize((28, 28))
        x += widthOfEachWord
        y1 += widthOfEachWord
        img = img.convert('L')
        a += 1
        x = np.asarray(img)
        vec = np.zeros((1, 784))
        k = 0
        zero = 0
        for i in range(28):
            for j in range(28):
                temp = x[i][j]
                if temp == 0:
                    zero += 1
                vec[0][k] = temp
                k += 1

        Theta1 = np.loadtxt('Theta1.txt')
        Theta2 = np.loadtxt('Theta2.txt')
        pred = predict(Theta1, Theta2, vec / 255)
        if zero == 784:
            result += ' '
        elif pred[0] > 9:
            dict = {10: 'A', 11: 'B', 12: 'C', 13: 'D', 14: 'E', 15: 'F', 16: 'G', 17: 'H', 18: 'I', 19: 'J', 20: 'K',
                    21: 'L', 22: 'M', 23: 'N', 24: 'O', 25: 'P', 26: 'Q', 27: 'R', 28: 'S', 29: 'T', 30: 'U', 31: 'V',
                    32: 'W', 33: 'X', 34: 'Y', 35: 'Z', 36: 'a', 37: 'b', 38: 'd', 39: 'e', 40: 'f', 41: 'g', 42: 'h',
                    43: 'n', 44: 'q', 45: 'r', 46: 't'}
            result += dict[pred[0]]
        else:
            result += (str)(pred[0])
    l1 = Label(window, text="Medicine:- " + result, font=('Algerian', 20), fg="green", anchor='center')
    l1.place(x=435, y=270)


lastx, lasty = None, None


def drawGrid():
    for i in range(1, 10):
        j = i * 100
        cv.create_line(j, 0, j, j + widthOfEachWord, fill="white")


def clear_widget():
    global cv, cv1, l1
    cv.delete("all")
    l1.destroy()
    drawGrid()


def event_activation(event):
    global lastx, lasty
    cv.bind('<B1-Motion>', draw_lines)
    lastx, lasty = event.x, event.y


def draw_lines(event):
    global lastx, lasty
    x, y = event.x, event.y
    cv.create_line((lastx, lasty, x, y), width=14, fill='white', capstyle=ROUND, smooth=TRUE, splinesteps=12)
    lastx, lasty = x, y


cv = Canvas(window, width=10 * widthOfEachWord, height=140, bg='black')
cv.place(x=100, y=115)

L1 = Label(window, text="Prescription Recoginition", font=('Algerian', 25), fg="blue")
L1.place(x=380, y=0)

b1 = Button(window, text="1. Clear Canvas", font=('Algerian', 15), bg="orange", fg="black", command=clear_widget)
b1.place(x=345, y=45)

b2 = Button(window, text="2. Prediction", font=('Algerian', 15), bg="white", fg="red", command=MyProject)
b2.place(x=719, y=45)

drawGrid()
cv.bind('<Button-1>', event_activation)
window.geometry("1200x350")
window.mainloop()
