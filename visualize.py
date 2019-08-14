import matplotlib

matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib import patheffects, patches


def show_img(img, figsize=None, fig=None, ax=None):
    if not ax:
        fig, ax = plt.subplots(figsize=figsize)

    ax.imshow(img)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    return fig, ax


def draw_outline(obj, line_width):
    obj.set_path_effects(
        [
            patheffects.Stroke(linewidth=line_width, foreground='black'),
            patheffects.Normal(),
        ]
    )


def draw_rect(ax, box):
    patch = ax.add_patch(
        patches.Rectangle(box[:2], *box[-2:], fill=False, edgecolor='white', lw=2)
    )
    draw_outline(patch, 4)


def draw_text(ax, xy, txt, sz=14):
    text = ax.text(
        *xy, txt, verticalalignment='top', color='white', fontsize=sz, weight='bold'
    )
    draw_outline(text, 1)
