"""Native playback controls around the actual SuperTuxKart camera image."""

from __future__ import annotations

import time

from .playback import METHODS, annotated_frame


def choices(playback, field):
    if field == "method":
        return [
            (m, METHODS[m])
            for m in METHODS
            if m == "expert"
            or any(
                e == playback.track and method == m for e, method, _ in playback.catalog
            )
        ]
    if field == "track":
        titles = {"lighthouse": "Lighthouse", "snowtuxpeak": "Snow Tux Peak"}
        return [
            (track, titles.get(track, track.replace("_", " ").title()))
            for track in playback.tracks
        ]
    if playback.method == "expert":
        return [(0, "Expert reference")]
    return [
        (s, f"Training seed {s}")
        for s in sorted(
            {
                s
                for e, m, s in playback.catalog
                if e == playback.track and m == playback.method
            }
        )
    ]


def select(playback, field, value):
    config = {
        name: getattr(playback, name)
        for name in [
            "method",
            "train_seed",
            "track",
            "eval_seed",
        ]
    }
    config[field] = value
    if config["method"] != "expert":
        available = sorted(
            s
            for e, m, s in playback.catalog
            if e == config["track"] and m == config["method"]
        )
        if not available:
            config["method"] = "expert"
            config["train_seed"] = 0
            playback.start(**config)
            return
        if config["train_seed"] not in available:
            config["train_seed"] = available[0]
    playback.start(**config)


def show(playback, quit_after_steps=None):
    import tkinter as tk
    from tkinter import ttk

    from PIL import ImageTk

    root = tk.Tk()
    root.title("SuperTuxKart — trained policies")
    root.resizable(False, False)
    toolbar = ttk.Frame(root, padding=10)
    toolbar.pack(fill="x")
    picture = ttk.Label(root)
    picture.pack()
    actions = ttk.Frame(root, padding=10)
    actions.pack(fill="x")
    status = tk.StringVar()
    ttk.Label(root, textvariable=status, padding=(10, 0, 10, 10)).pack(anchor="w")
    fields = [
        ("method", "Policy", 25),
        ("train_seed", "Training run", 16),
        ("track", "Track", 18),
    ]
    selectors, labels = {}, {}
    paused, speed, photo = False, 1.0, None
    closing, last_step = False, time.perf_counter()

    def refresh_image():
        nonlocal photo
        photo = ImageTk.PhotoImage(annotated_frame(playback), master=root)
        picture.configure(image=photo)
        checkpoint = (
            "Expert reference"
            if playback.checkpoint_step is None
            else f"{playback.checkpoint_step:,} training steps"
        )
        status.set(
            f"{checkpoint}  ·  evaluation seed {playback.eval_seed}  ·  "
            "Deterministic policy; collection noise is off"
        )

    def refresh_choices():
        for name, _, _ in fields:
            options = choices(playback, name)
            labels[name] = {label: value for value, label in options}
            selectors[name].configure(values=list(labels[name]))
            selectors[name].set(dict(options)[getattr(playback, name)])

    def change(name):
        nonlocal last_step
        try:
            select(playback, name, labels[name][selectors[name].get()])
            refresh_choices()
            refresh_image()
            last_step = time.perf_counter()
        except (ValueError, RuntimeError, OSError) as exc:
            status.set(str(exc))

    for i, (name, title, width) in enumerate(fields):
        ttk.Label(toolbar, text=title).grid(row=0, column=i, sticky="w", padx=(0, 8))
        box = ttk.Combobox(toolbar, state="readonly", width=width)
        box.grid(row=1, column=i, padx=(0, 8), pady=(4, 0))
        box.bind("<<ComboboxSelected>>", lambda event, field=name: change(field))
        selectors[name] = box

    def play_pause(event=None):
        nonlocal paused, last_step
        if event is not None and event.widget.winfo_class() == "TCombobox":
            return
        if playback.done:
            playback.restart()
            paused = False
            refresh_image()
        else:
            paused = not paused
        play.configure(text="Play" if paused else "Pause")
        last_step = time.perf_counter()

    def advance(event=None):
        nonlocal paused, last_step
        if event is not None and event.widget.winfo_class() == "TCombobox":
            return
        paused = True
        play.configure(text="Play")
        playback.step()
        refresh_image()
        last_step = time.perf_counter()

    def restart(event=None):
        nonlocal last_step
        playback.restart()
        refresh_image()
        last_step = time.perf_counter()

    def change_speed(event=None):
        nonlocal speed, last_step
        speed = float(rate.get().replace("×", ""))
        last_step = time.perf_counter()

    play = ttk.Button(actions, text="Pause", command=play_pause)
    play.pack(side="left", padx=(0, 8))
    ttk.Button(actions, text="Step →", command=advance).pack(side="left", padx=(0, 8))
    ttk.Button(actions, text="Restart", command=restart).pack(side="left", padx=(0, 16))
    ttk.Label(actions, text="Playback speed").pack(side="left", padx=(0, 8))
    rate = ttk.Combobox(
        actions, state="readonly", values=["0.25×", "0.5×", "1×", "2×", "4×"], width=7
    )
    rate.set("1×")
    rate.pack(side="left")
    rate.bind("<<ComboboxSelected>>", change_speed)
    ttk.Label(actions, text="Space: pause    →: step    R: restart").pack(side="right")

    def close(event=None):
        nonlocal closing
        closing = True
        root.destroy()

    def tick():
        nonlocal last_step
        if closing:
            return
        now = time.perf_counter()
        if not paused and not playback.done and now - last_step >= 0.1 / speed:
            playback.step()
            refresh_image()
            last_step = now
        if quit_after_steps is not None and playback.steps >= quit_after_steps:
            close()
            return
        root.after(10, tick)

    root.protocol("WM_DELETE_WINDOW", close)
    root.bind("<space>", play_pause)
    root.bind("<Right>", advance)
    root.bind("r", restart)
    root.bind("<Escape>", close)
    refresh_choices()
    refresh_image()
    root.after(10, tick)
    try:
        root.mainloop()
    finally:
        if not closing:
            root.destroy()
