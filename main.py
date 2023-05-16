import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage.filters import gaussian
from skimage.segmentation import active_contour
from multiprocessing import Pool
import logging
import concurrent.futures
from sklearn.cluster import KMeans

logging.basicConfig(filename='app.log', filemode='w', format='%(asctime)s - %(message)s', level=logging.INFO)


def read_image(in_file):
    image = plt.imread(in_file)
    return image


def detect_wound(image, out_file):
    # Convert RGB to grayscale.
    gray = rgb2gray(image)

    # Apply Gaussian filter.
    blur = gaussian(gray, 3)

    # Initialize snake.
    fig, axs = plt.subplots(10, 10, figsize=(20, 20))
    fig.subplots_adjust(hspace=0.5)
    params = []
    for i in range(10):
        for j in range(10):
            s = np.linspace(0, 2 * np.pi, 400)
            x = (image.shape[1] / 2) + (image.shape[1] / 4) * np.cos(s)
            y = (image.shape[0] / 2) + (image.shape[0] / 4) * np.sin(s)
            init = np.array([x, y]).T

            # Generate random parameters.
            alpha = np.random.uniform(0.01, 0.05)
            beta = np.random.uniform(1.0, 5.0)
            gamma = np.random.uniform(0.001, 0.01)

            params.append((blur, init, alpha, beta, gamma))

            logging.info(f'x={x}, y={y}')

    with Pool() as p:
        logging.info('Pool() called')
        results = p.starmap(run_snake_with_params, params)

    for i in range(10):
        for j in range(10):
            ax = axs[i][j]
            ax.imshow(image)
            snake_array = np.array(results[i * 10 + j][0])
            ax.plot(snake_array[:, 0], snake_array[:, 1], '-r', lw=3)
            ax.set_title(
                f'a={results[i * 10 + j][1]:.3f}, b={results[i * 10 + j][2]:.3f}, g={results[i * 10 + j][3]:.3f}')

            area = calculate_area(snake_array)
            ax.text(0.5, -0.1, f'Area: {area:.2f}', size=12, ha="center", transform=ax.transAxes)

    plt.savefig(out_file)


def run_snake_with_params(blur, init, alpha, beta, gamma):
    snake = active_contour(blur, init, alpha=alpha, beta=beta, gamma=gamma)
    return [snake, alpha, beta, gamma]


def calculate_area(snake_array):
    x, y = snake_array.T
    return 0.5 * np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def run_snake_with_params_clustered_wounds(blur, labels, mask, init, alpha, beta, gamma):
    snake = active_contour((blur * mask).reshape(labels.shape), init,
                           alpha=alpha, beta=beta, gamma=gamma)
    return snake


def detect_wound_cluster(image, out_file):
    # Convert RGB to grayscale.
    gray = rgb2gray(image)

    # Apply Gaussian filter.
    blur = gaussian(gray, 3)

    # Perform K-means clustering.
    kmeans = KMeans(n_clusters=2).fit(blur.reshape(-1, 1))
    labels = kmeans.labels_.reshape(blur.shape)

    # Initialize snake.
    fig, axs = plt.subplots(2, figsize=(5, 10))
    params = []
    for i in range(2):
        s = np.linspace(0, 2*np.pi, 400)
        x = (image.shape[1] / 2) + (image.shape[1] / 4) * np.cos(s)
        y = (image.shape[0] / 2) + (image.shape[0] / 4) * np.sin(s)
        init = np.array([x, y]).T

        # Generate random parameters.
        alpha = np.random.uniform(0.01, 0.05)
        beta = np.random.uniform(1.0, 5.0)
        gamma = np.random.uniform(0.001, 0.01)

        params.append((blur.reshape(labels.shape) * (labels == i), labels == i,
                       init, alpha, beta, gamma))

    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = [executor.submit(run_snake_with_params_clustered_wounds,
                                   blur,
                                   labels,
                                   params[i][0],
                                   params[i][1],
                                   params[i][2],
                                   params[i][3],
                                   params[i][4]) for i in range(len(params))]

        for i in range(len(results)):
            ax = axs[i]
            ax.imshow(image)
            snake_array = results[i].result()
            ax.plot(snake_array[:, 0], snake_array[:, 1], '-r', lw=3)
            ax.set_title(f'Wound {i+1}')

            area = calculate_area(snake_array)
            ax.text(0.5,-0.1,f'Area: {area:.2f}', size=12, ha="center", transform=ax.transAxes)

    plt.savefig(out_file)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    image = read_image('images/image1.jpg')
    detect_wound(image, 'images/image1_out_cluster.jpg')
    logging.info('Image processing completed')

