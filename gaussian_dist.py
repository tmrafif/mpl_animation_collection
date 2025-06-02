import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.animation import FuncAnimation

## Functions 
def gaussianPDF(x, mu=0, sigma=1):
    return stats.norm.pdf(x, mu, sigma)

def gaussianCDF(x, mu=0, sigma=1):
    return stats.norm.cdf(x, mu, sigma)

def gaussianPPF(p, mu=0, sigma=1):
    return stats.norm.ppf(p, mu, sigma)

def format_axes(fig):
    label_style = dict(fontsize=12, usetex=True)
    for i, ax in enumerate(fig.axes):
        ax.spines[:].set_color('black')
        ax.tick_params(length=3)

    ax = fig.axes

    ax[0].set_xlim(-5, 5)
    ax[0].set_ylim(0, 0.44)
    ax[0].set_xlabel(r'$x$', **label_style)
    ax[0].set_ylabel(r'$f(x)$', **label_style)
    ax[0].legend(frameon=True)

    ax[1].set_xlim(-5, 5)
    ax[1].set_ylim(0, 1.1)
    ax[1].set_xlabel(r'$x$', **label_style)
    ax[1].set_ylabel(r'$\Phi(x)$', **label_style)
    ax[1].legend(frameon=True, loc='lower right')

    ax[2].set_xlim(0, 1)
    ax[2].set_ylim(-4.2, 4.2)
    ax[2].set_xlabel(r'$p$', **label_style)
    ax[2].set_ylabel(r'$\Phi^{-1}(p)$', **label_style)
    ax[2].legend(frameon=True)

# Initialize the data
x = np.linspace(-5, 5, 1000)
p = np.linspace(1e-4, 1-1e-4, 1000)

# Set up the figure and axes
plt.style.use(['seaborn-v0_8-paper', 'seaborn-v0_8-whitegrid'])
# plt.rcParams['text.usetex'] = True
fig = plt.figure(figsize=(8, 6))
gs = GridSpec(2, 2, figure=fig)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[1, 0])
ax3 = fig.add_subplot(gs[0:, 1])
ax = fig.axes

# Plot the PDF
ax[0].plot(x, gaussianPDF(x), label='PDF')

# Plot the CDF
ax[1].plot(x, gaussianCDF(x), color='r', label='CDF')

# Plot the PPF
ax[2].plot(p, gaussianPPF(p), color='k', label='PPF')

format_axes(fig)

## Animation
fps = 60
gif_length = 2  # seconds

# styling
dot_style = dict(marker='o', markersize=6, alpha=0.5)
line_style = dict(lw=1, linestyle='--', alpha=0.7)

# plot 1
fill_plot = ax[0].fill_between([], [], color="red", alpha=0.4)
line_pdf_plot, = ax[0].plot([], [], color='k', **line_style)

# plot 2
line_x_cdf_plot, = ax[1].plot([], [], color='k', **line_style)
line_y_cdf_plot, = ax[1].plot([], [], color='r', **line_style)
dot_cdf_plot, = ax[1].plot([], [], color='r', **dot_style)

# plot 3
line_x_ppf_plot, = ax[2].plot([], [], color='k', **line_style)
line_y_ppf_plot, = ax[2].plot([], [], color='r', **line_style)
dot_ppf_plot, = ax[2].plot([], [], color='k', **dot_style)

def update_data(frame):
    t = frame / fps
    omega = 0.5   # cycles per second
    fill_to = 2*np.sin(2*np.pi*omega*t)  # Vary the fill_to value
    x_fill = np.linspace(-5, fill_to, 1000)

    # Update PDF plot
    fill_plot.set_data(x_fill, gaussianPDF(x_fill), 0)
    line_pdf_plot.set_data([fill_to, fill_to], [0, gaussianPDF(fill_to)])
    
    # Update CDF plot
    line_x_cdf_plot.set_data([fill_to, fill_to], [0, gaussianCDF(fill_to)])
    line_y_cdf_plot.set_data([-5, fill_to], [gaussianCDF(fill_to), gaussianCDF(fill_to)])
    dot_cdf_plot.set_data([fill_to], [gaussianCDF(fill_to)])
    
    # Update PPF plot
    line_x_ppf_plot.set_data([0, gaussianCDF(fill_to)], [fill_to, fill_to])
    line_y_ppf_plot.set_data([gaussianCDF(fill_to), gaussianCDF(fill_to)], [-5, fill_to])
    dot_ppf_plot.set_data([gaussianCDF(fill_to)], [fill_to])

    return dot_cdf_plot, line_x_cdf_plot, line_y_cdf_plot, dot_ppf_plot, fill_plot

animation = FuncAnimation(
    fig=fig,
    func=update_data,
    frames=fps*gif_length,
    interval=1000/fps,
)

# Save the animation as a GIF
plt.tight_layout()
animation.save('output/gaussian_dist.gif', fps=fps)
print('Animation saved!')

plt.show()