from tkinter import *
from tkinter import ttk
import cp
import random

root = Tk()
root.title("CP Generator")

# set true to label vertices
show_labels = False

# Main Frame
mainframe = ttk.Frame(root, padding="3 3 12 12")

def draw_cp(cp, canvas, side_length=500):
    cp.normalize()
    for f in cp.folds:
        # red for mountain, blue for valley
        if f.type == 0:
            color = "red"
        elif f.type == 1:
            color = "blue"
        else:
            color = "black"
        canvas.create_line(f.v1.x*500, f.v1.y*500, f.v2.x*500, f.v2.y*500, fill=color)
# Create a Frame to contain the Canvas
frame = Frame(root, borderwidth=2, relief="ridge")
frame.pack(padx=20, pady=20)  # Adjust padx and pady for the desired padding
frame.pack(side="left", fill="both", expand=True)

def label_vertices(cp, canvas, side_length=500):
    for v in cp.vertices:
        # shift the label a bit so it's within the canvas bounds
        shift = 0.01
        if v.x < 0.5:
            x = v.x*500 + shift*500
        else:
            x = v.x*500 - shift*500
        if v.y < 0.5:
            y = v.y*500 + shift*500
        else:
            y = v.y*500 - shift*500
        canvas.create_text(x, y, text=str(cp.vertices.index(v)))

# Create the Canvas widget inside the Frame
h = 500
w = 500
canvas = Canvas(frame, width=w, height=h, background="white", borderwidth=0, highlightthickness=0)
canvas.pack()
# Make a CP with random vertices
cp = cp.CreasePattern()
cp.side = 500
cp.add_square_vertices()
for i in range(8):
    cp.add_random_vertex()
cp.push_to_edge(10)
cp.triangulate()
cp.evenize_vertices()
cp.remove_edge_folds()
cp.optimize()

cp.assign_mv()
# Draw the CP
draw_cp(cp, canvas)
text_var = StringVar()
text_var.set("5")



def make_cp():
    text_var.set(text_input.get())
    # check if text is a number
    try:
        num = int(text_input.get())
    except ValueError:
        num = 0
    cp.clear()
    canvas.delete("all")
    cp.side = 500
    for i in range(num):
        cp.add_random_vertex()
    cp.add_square_vertices()
    cp.push_to_edge(20)
    cp.triangulate()
    cp.evenize_vertices()
    cp.remove_edge_folds()
    if show_labels:
        label_vertices(cp, canvas)
    draw_cp(cp, canvas)

def optimize_cp():
    res = cp.optimize()
    canvas.delete("all")
    if show_labels:
        label_vertices(cp, canvas)
    draw_cp(cp, canvas)
    update_text(res.message)

def assign_mv():
    res = cp.assign_mv()
    canvas.delete("all")
    if show_labels:
        label_vertices(cp, canvas)
    draw_cp(cp, canvas)
    if res == []:
        update_text("no possible MV assignment")
        for fold in cp.folds:
            fold.type = -1
    else:
        update_text("Successfully assigned MV")

def update_text(res):
    text_label.config(text=res)

def export_svg():
    cp.export_svg("cp.svg")

# Create label
label = Label(root, text="Number of points")
label.pack()

# Create an Entry widget for text input
text_input = Entry(root)
text_input.pack()

# Create a button for generating the CP
button1 = Button(root, text="Generate CP", command=make_cp)
button1.pack()

# Create a button for optimizing the CP
button2 = Button(root, text="Optimize for Kawasaki's Theorem", command=optimize_cp)
button2.pack()

# Create a button for assigning mountain and valley folds
button3 = Button(root, text="Assign MV", command=assign_mv)
button3.pack()

# Create a label for displaying text
initial_text = ""
text_label = Label(root, text=initial_text)
text_label.pack()

# Create a button for exporting the svg
button3 = Button(root, text="Export svg", command=export_svg)
button3.pack()

# Start
root.mainloop()