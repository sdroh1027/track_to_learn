from matplotlib import pyplot

def plot_tensor(tensor, square):

    # plot all 64 maps in an 8x8 squares
    ix = 1
    for _ in range(square):
        for _ in range(square):
            # specify subplot and turn of axis
            ax = pyplot.subplot(square, square, ix) # ixth one among square * squate subplots
            ax.set_xticks([])
            ax.set_yticks([])
            # plot filter channel in grayscale
            pyplot.imshow(tensor[ix - 1, :, : ], cmap='gray')
            ix += 1
    # show the figure
    pyplot.show()