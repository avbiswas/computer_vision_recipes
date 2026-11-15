"""Play the real FireflyRoom. Arrows move the lamp; the world ticks continuously.
keys: arrows = move lamp, r = reset, q = quit
"""
import matplotlib.pyplot as plt
from env import FireflyRoom, DIR_NAMES

env = FireflyRoom(res=160)
held = {"d": 0}
fig, ax = plt.subplots(figsize=(6, 6))
im = ax.imshow(env.render(), interpolation="nearest")
ax.axis("off")
ax.set_title("arrows: move lamp   r: reset")


def on_press(e):
    if e.key in DIR_NAMES:
        held["d"] = DIR_NAMES.index(e.key)
    elif e.key == "r":
        env.reset()


def on_release(e):
    if e.key in DIR_NAMES:
        held["d"] = 0


def tick():
    im.set_data(env.step(held["d"]))
    fig.canvas.draw_idle()


fig.canvas.mpl_connect("key_press_event", on_press)
fig.canvas.mpl_connect("key_release_event", on_release)
timer = fig.canvas.new_timer(interval=50)
timer.add_callback(tick)
timer.start()
plt.show()
