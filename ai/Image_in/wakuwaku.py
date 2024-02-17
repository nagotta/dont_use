import tkinter as tk
import tkinter as tk
from tkinter.constants import N
import tkinter.filedialog as fd
import PIL.Image
import PIL.ImageTk
#アプリのウィンドウを作る
root = tk.Tk()
root.geometry("400x400")

btn = tk.Button(root, text="ファイルを開く", command = openFile)
imageLabel = tk.Label()
btn.pack()
imageLabel.pack()

#予測結果を表示するラベル
textLabel = tk.Label(text="手書きの数字を認識します！")
textLabel.pack()

tk.mainloop()