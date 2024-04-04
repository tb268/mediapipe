import tkinter as tk

class ButtonTextGroup(tk.Frame):
    def __init__(self, parent, button_text, label_text):
        super().__init__(parent)

        self.label_text = label_text

        # ラベルとボタンを作成し、親フレームに配置する
        self.label = tk.Label(self, text=self.label_text)
        self.label.pack(side=tk.LEFT, padx=(5, 0))

        self.button = tk.Button(self, text=button_text, command=self.update_label)
        self.button.pack(side=tk.LEFT)

    def update_label(self):
        # ボタンがクリックされたときにラベルのテキストを変更する
        self.label.config(text="Button clicked: " + self.label_text)

class MainApplication(tk.Tk):
    def __init__(self):
        super().__init__()
        self.geometry("400x100")
        self.title("Button and Text Group")

        # Frame 1
        frame1 = tk.Frame(self)
        frame1.pack(side=tk.LEFT, padx=5)

        # Frame 2
        frame2 = tk.Frame(self)
        frame2.pack(side=tk.LEFT, padx=5)

        # Create ButtonTextGroup instances and pack them into their respective frames
        ButtonTextGroup(frame1, "Button 1", "Text 1").pack(side=tk.TOP)
        ButtonTextGroup(frame2, "Button 2", "Text 2").pack(side=tk.TOP)
        ButtonTextGroup(frame1, "Button 3", "Text 3").pack(side=tk.TOP)

if __name__ == "__main__":
    app = MainApplication()
    app.mainloop()
