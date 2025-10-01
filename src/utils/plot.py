import matplotlib
import matplotlib.pylab as plt
import numpy as np

import logging

mpl_logger = logging.getLogger('matplotlib')
mpl_logger.setLevel(logging.WARNING)

matplotlib.use('Agg')


def plot_spectrogram_to_numpy(spectrogram):
    fig, ax = plt.subplots(figsize=(10, 2))
    im = ax.imshow(spectrogram, aspect='auto', origin='lower', interpolation='none')
    plt.colorbar(im, ax=ax)
    plt.xlabel('Frames')
    plt.ylabel('Channels')
    plt.tight_layout()

    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    w, h = fig.canvas.get_width_height()
    data = data.reshape((h, w, 4))[:, :, :3]  # RGB only
    plt.close(fig)

    return data


def plot_alignment_to_numpy(alignment, info=None):
    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(alignment.transpose(), aspect='auto', origin='lower', interpolation='none')
    fig.colorbar(im, ax=ax)
    xlabel = 'Decoder timestep'
    if info is not None:
        xlabel += '\n\n' + info
    plt.xlabel(xlabel)
    plt.ylabel('Encoder timestep')
    plt.tight_layout()

    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    w, h = fig.canvas.get_width_height()
    data = data.reshape((h, w, 4))[:, :, :3]  # RGB only
    plt.close(fig)

    return data
