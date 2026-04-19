# sample_viewer.py
import sys, json, base64, io
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button


def load_frames(paths):
    frames = []
    for p in paths:
        with open(p, "r") as f:
            for line in f:
                line = line.strip()
                if line: frames.append(json.loads(line))
    return sorted(frames, key=lambda x: x["epoch"])


def decode(frame):
    return np.array(Image.open(io.BytesIO(base64.b64decode(frame["grid_b64"]))))


def view(frames):
    if not frames: raise RuntimeError("No frames to view")
    print(f"Loaded {len(frames)} frames (epochs {frames[0]['epoch']} -> {frames[-1]['epoch']})")

    fig, ax = plt.subplots(figsize=(8, 9))
    plt.subplots_adjust(bottom=0.18)
    handle = ax.imshow(decode(frames[0]))
    ax.axis("off")
    title = ax.set_title(f"Epoch {frames[0]['epoch']} (1/{len(frames)})")

    ax_slider = plt.axes([0.12, 0.08, 0.66, 0.03])
    slider = Slider(ax_slider, "frame", 0, len(frames) - 1, valinit=0, valstep=1)

    ax_play = plt.axes([0.82, 0.075, 0.1, 0.04])
    btn = Button(ax_play, "Play")

    state = {"playing": False}

    def render(i):
        handle.set_data(decode(frames[i]))
        title.set_text(f"Epoch {frames[i]['epoch']} ({i+1}/{len(frames)})")
        fig.canvas.draw_idle()

    def on_slide(val): render(int(slider.val))
    slider.on_changed(on_slide)

    def on_key(event):
        i = int(slider.val)
        if event.key == "right": slider.set_val(min(i + 1, len(frames) - 1))
        elif event.key == "left": slider.set_val(max(i - 1, 0))
        elif event.key == " ": toggle()

    def toggle(_=None):
        state["playing"] = not state["playing"]
        btn.label.set_text("Pause" if state["playing"] else "Play")
        if state["playing"]: step()

    def step():
        if not state["playing"]: return
        i = int(slider.val)
        slider.set_val(0 if i == len(frames) - 1 else i + 1)
        fig.canvas.start_event_loop(0.08)  # ~12 fps
        step()

    btn.on_clicked(toggle)
    fig.canvas.mpl_connect("key_press_event", on_key)
    plt.show()


if __name__ == "__main__":
    args = sys.argv[1:] or [
        "outputs/samples_01.jsonl",
        "outputs/samples_02.jsonl",
        "outputs/samples_03.jsonl",
    ]
    view(load_frames(args))