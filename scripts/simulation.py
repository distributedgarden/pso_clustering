import numpy as np
import argparse
import matplotlib.pyplot as plt
import pandas as pd
import time
import logging
import tqdm

from typing import Tuple

from sklearn.cluster import KMeans
from sklearn.datasets import load_iris, load_wine, fetch_openml
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


logging.basicConfig(level=logging.INFO)


def plot_clusters(
    data: np.ndarray,
    true_labels: np.ndarray,
    predicted_labels: np.ndarray,
    centers: np.ndarray,
    title: str,
) -> None:
    """
    Description:
    - Create a scatter plot of clustered data points and cluster centers
    - Use PCA to reduce the dimensions of the data to 3D for visualization
    """
    pca = PCA(n_components=3)
    data_3d = pca.fit_transform(data)
    fig = plt.figure()

    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(
        data_3d[:, 0],
        data_3d[:, 1],
        data_3d[:, 2],
        c=predicted_labels,
        cmap="viridis",
        marker="o",
        edgecolors="k",
    )

    centers_3d = pca.transform(centers)
    ax.scatter(
        centers_3d[:, 0],
        centers_3d[:, 1],
        centers_3d[:, 2],
        c="red",
        marker="x",
        s=100,
        label="Cluster Centers",
    )

    plt.title(title)
    plt.legend()
    plt.savefig("data/" + title + ".png")
    plt.close()


def plot_aggregated_measurements(
    dataset_names: list[str],
    error_rates: list[float],
    computation_times: list[float],
    best_obj_values: list[float],
    avg_obj_values: list[float],
    worst_obj_values: list[float],
    actual_clusters: list[int],
    predicted_clusters: list[int],
) -> None:
    """
    Description:
    - Plot the error rates, computation times, and fitness values
    - Aggregate the results for all datasets
    """
    # Plot error rates
    plt.figure()
    plt.bar(dataset_names, error_rates)
    plt.title("Error Rates")
    plt.ylabel("Error Rate (%)")
    plt.savefig("data/error_rates.png")

    # Plot computation times
    plt.figure()
    plt.bar(dataset_names, computation_times)
    plt.title("Computation Times")
    plt.ylabel("Computation Time (s)")
    plt.savefig("data/computation_times.png")

    # Plot fitness values
    fig, ax = plt.subplots()
    x_labels = dataset_names
    x = np.arange(len(x_labels))
    ax.bar(x - 0.2, best_obj_values, width=0.2, label="Best")
    ax.bar(x, avg_obj_values, width=0.2, label="Average")
    ax.bar(x + 0.2, worst_obj_values, width=0.2, label="Worst")
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_xlabel("Dataset")
    ax.set_ylabel("Objective Function Value")
    ax.set_yscale("log")  # Change y-axis scale to logarithmic
    ax.legend()
    ax.set_title("Fitness Values")
    plt.savefig("data/fitness_values.png")
    plt.close(fig)

    # Plot actual and predicted clusters
    fig, ax = plt.subplots()
    x_labels = dataset_names
    x = np.arange(len(x_labels))
    ax.bar(x - 0.2, actual_clusters, width=0.4, label="Actual")
    ax.bar(x + 0.2, predicted_clusters, width=0.4, label="Predicted")
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_xlabel("Dataset")
    ax.set_ylabel("Number of Clusters")
    ax.legend()
    ax.set_title("Actual vs Predicted Clusters")
    plt.savefig("data/actual_vs_predicted_clusters.png")
    plt.close(fig)


def error_rate(true_labels: np.ndarray, predicted_labels: np.ndarray) -> float:
    """
    Calculate the error rate (ER) between true labels and predicted labels of objects.

    The error rate is defined as the percentage of misplaced pairs of objects relative to the total number of all pairs.
    A pair is considered misplaced if the two objects are in the same cluster in one labeling but not in the other, or vice versa.
    This method of measuring solution quality is particularly useful in clustering tasks where the objective is to minimize
    the number of incorrectly grouped object pairs.

    Parameters:
    - true_labels (np.ndarray): An array of true labels for each object.
    - predicted_labels (np.ndarray): An array of predicted labels for each object,
        corresponding to the true labels.

    Returns:
    - float: The error rate (ER) as a percentage, indicating the proportion of object pairs that are incorrectly labeled
        in the predicted labels compared to the true labels.

    Formula:
    ER = \[ \frac{ ( \sum_{i=1}^{n-1} \sum_{l=i+1}^{n} |A_{il} - B_{il}| ) } { \frac{n(n-1)}{2} }\] \times 100
    where A_{il} and B_{il} are indicators of whether objects i and l are in the same cluster according to true labels and predicted labels,
    respectively, and n is the total number of objects.

    Example:
    Given a set of true labels and predicted labels for a dataset, the function computes the ER, providing insights into the
    quality of the clustering performed by the predictive model in terms of how well it matches the true clustering.
    """
    n = len(true_labels)
    error_sum = 0
    for i in range(n - 1):
        for l in range(i + 1, n):
            error_sum += abs(
                int(true_labels[i] == true_labels[l])
                - int(predicted_labels[i] == predicted_labels[l])
            )
    err = (error_sum / (n * (n - 1) / 2)) * 100

    return err


def load_dataset(name: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Description:
    - Load a dataset from the scikit-learn library
    - Clean the data if it is a pandas DataFrame
    """
    if name == "iris":
        dataset = load_iris()
    elif name == "wine":
        dataset = load_wine()
    elif name == "thyroid":
        dataset = fetch_openml(name="thyroid-new")
    elif name == "cmc":
        dataset = fetch_openml(name="cmc")
    elif name == "glass":
        dataset = fetch_openml(name="glass")
    else:
        raise ValueError("Invalid dataset name")

    data = dataset.data
    labels = dataset.target
    logging.info(
        f"Loaded {name} dataset with {data.shape[0]} samples and {data.shape[1]} features"
    )

    # If the dataset is a pandas DataFrame, clean the data
    if isinstance(data, pd.DataFrame):
        for col in data.columns:
            if pd.api.types.is_categorical_dtype(data[col]):
                dummies = pd.get_dummies(data[col], prefix=col)
                data = pd.concat([data.drop(col, axis=1), dummies], axis=1)
            else:
                for i, value in enumerate(data[col]):
                    if isinstance(value, str) and value.isdigit():
                        data.at[i, col] = float(value)
        # Remove non-float types
        # data = data.select_dtypes(include='float64')

        # Check for missing values and handle them (fill with mean)
        if data.isnull().values.any():
            data = data.fillna(data.mean())

        # Convert the cleaned DataFrame back to a NumPy array
        data = data.to_numpy()

    return data, labels


def custom_kmeans(
    data: np.ndarray, initial_centroids: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Description:
    - Perform k-means clustering using the given initial centroids
    """
    kmeans = KMeans(
        n_clusters=initial_centroids.shape[0], init=initial_centroids, n_init=1
    ).fit(data)
    clusters = kmeans.predict(data)
    centers = kmeans.cluster_centers_

    return clusters, centers


def calculate_intra_cluster_fitness(
    data: np.ndarray, clusters: np.ndarray, centers: np.ndarray
) -> float:
    """
    Computes the fitness value of a particle (solution) in the context of a clustering problem,
    as per the definition by Kennedy and Eberhart (1995). The fitness value represents the quality
    of a clustering solution, with a lower fitness value indicating a better solution.

    The fitness value is calculated as the sum of weighted distances between data points and their
    respective cluster centers. This approach aims to minimize the intra-cluster distances, thus
    enhancing the compactness of clusters.
    1. f_p^t = \sigma_p^t = \Sigma_{k=1}^{K_p} \Sigma_{i=1}^{n} w_{ik}^{pt} D(o_i, z_k^{pt}) (equation 7)

    Parameters:
    - data (np.ndarray): A 2D array where each row represents a data point and each column a feature.
    - clusters (np.ndarray): An array of cluster assignments for each data point. The length of this
      array is equal to the number of rows in `data`.
    - centers (np.ndarray): A 2D array where each row represents the coordinates of a cluster center.

    Returns:
    - float: The fitness value of the clustering solution. Lower values indicate tighter, more
      coherent clusters, which is desirable in clustering tasks.

    Example:
    This function is typically used within particle swarm optimization algorithms to evaluate
    and guide the search process towards optimal clustering solutions. It quantifies how well
    a given particle (solution) organizes the data into clusters, based on the current positions
    of cluster centers and the assignment of data points to clusters.

    Note:
    The fitness function plays a crucial role in evolutionary algorithms for clustering,
    where it provides a measure of solution quality that influences the selection and
    update of particles' positions in the solution space.
    """
    N = data.shape[0]
    fit = 0
    for i in range(N):
        cluster_idx = clusters[i]
        centroid = centers[cluster_idx]

        fit += np.sum((data[i] - centroid) ** 2)

    return fit


def distance_matrix(centers: np.ndarray) -> np.ndarray:
    """
    Description:
    - Calculate the distance matrix between the given cluster centers
    """
    num_centers = centers.shape[0]
    dist_matrix = np.zeros((num_centers, num_centers))
    for i in range(num_centers):
        for j in range(num_centers):
            dist_matrix[i, j] = np.sum((centers[i] - centers[j]) ** 2)

    return dist_matrix


def calculate_adaptive_cluster_fitness(
    data: np.ndarray, clusters: np.ndarray, centers: np.ndarray
) -> float:
    """
    Computes the fitness of a clustering solution using a specific fitness function
    designed for clustering problems where the number of clusters is not predefined.
    1. f_p^t = \sigma_p^t-min_{ k\neq1 } D(z_k^{pt}, z_l^{pt}) (equation 8)

    This function evaluates the quality of a given clustering solution based on two criteria:
    1. The total intra-cluster distance (sum of squared distances from each point to its cluster center),
    2. The minimum inter-cluster distance (the smallest distance between any two cluster centers).

    The fitness score is calculated as the total intra-cluster distance minus the minimum inter-cluster distance.
    A lower fitness score indicates a better clustering solution, with tighter clusters that are farther apart.

    Parameters:
    - data (np.ndarray): The dataset being clustered, where rows represent samples and columns represent features.
    - clusters (np.ndarray): An array indicating the cluster assignment of each sample in the dataset.
    - centers (np.ndarray): The coordinates of the cluster centers.

    Returns:
    - float: The fitness score of the clustering solution.

    Notes:
    - This fitness function is particularly suitable for evolutionary algorithms where the number of clusters is not fixed in advance.
    - The fitness score aims to balance the compactness of clusters (through intra-cluster distance) against
        the separation between clusters (via inter-cluster distance), facilitating the identification of solutions that achieve a good trade-off between these two aspects.

    Example:
    This function can be used within an optimization loop where clustering solutions are iteratively improved.
    By calculating the fitness of each solution, the algorithm can identify and retain higher-quality clusterings over time.
    """
    N = data.shape[0]
    num_centers = centers.shape[0]

    # Make a copy of clusters and centers to prevent modification of the original inputs
    clusters_copy = np.copy(clusters)
    centers_copy = np.copy(centers)

    # Calculate rho_p for each center
    rho = np.zeros(num_centers)
    for i in range(N):
        cluster_idx = clusters_copy[i]
        centroid = centers_copy[cluster_idx]
        rho[cluster_idx] += np.sum((data[i] - centroid) ** 2)

    # Calculate the minimum inter-cluster distance
    dist_matrix = distance_matrix(centers_copy)
    min_inter_cluster_dist = np.min(dist_matrix + np.diag(np.full(num_centers, np.inf)))

    # Compute fitness value
    fitness = np.sum(rho) - min_inter_cluster_dist

    return fitness


def move_particle(
    current_particle: np.ndarray,
    personal_best_particle: np.ndarray,
    global_best_particle: np.ndarray,
) -> np.ndarray:
    """
    Description:
    Update the position of a particle p on dimension K according to a modified PSO rule.

    This function is intended to model the behavior of particle movement in a discrete
    space, specifically for updating the number of clusters `K_{p}^{t+1}` in a clustering
    problem. It applies a weighted combination of the current position, the personal best
    position, and the global best position to compute a new position, which is then adjusted
    by rounding and ensuring the value is at least 1, to maintain a valid number of clusters.
    1. K_{p}^{t+1} = max(round(vK_{p}^{t+1}+K_{p}^{t}),1) (equation 12)

    Parameters:
    - current_particle (np.ndarray): The current number of clusters for particle p.
    - personal_best_particle (np.ndarray): The personal best number of clusters for particle p.
    - global_best_particle (np.ndarray): The global best number of clusters across all particles.

    Returns:
    - np.ndarray: The updated number of clusters for particle p for the next iteration, with
                  the constraint that there is at least one cluster.

    Note:
    The actual implementation provided does not directly apply the `max` and `round` operations
    as per the specified update rule in equation 12. To fully conform to equation 12, modifications
    to this function are required to incorporate these steps, ensuring that the output reflects
    the discrete nature of the number of clusters in a clustering problem.
    """
    w = 0.9
    r1 = np.random.rand()
    r2 = np.random.rand()
    new_particle = (
        w * current_particle
        + r1 * (personal_best_particle - current_particle)
        + r2 * (global_best_particle - current_particle)
    )

    return new_particle


def move_particle_zkj(
    current_zkj: np.ndarray, personal_best_zkj: np.ndarray, global_best_zkj: np.ndarray
) -> np.ndarray:
    """
    Description:
    Update the position of a particle's cluster center in a specific dimension.

    This function updates the position `z_{kj}^{p, t+1}` of a particle's cluster center
    in the `z_{kj}`-th dimension for the next iteration `t+1`. The update is based on the
    current position `z_{kj}^{pt}`, the particle's personal best position,
    and the global best position in that dimension, according to the
    Particle Swarm Optimization (PSO) algorithm's movement strategy.
    1. z_{kj}^{p, t+1} = vz_{kj}^{p, t+1} + z_{kj}^{pt} (equation 13)

    Parameters:
    - current_zkj (np.ndarray): The current position of the particle's `z_{kj}`-th dimension cluster center.
    - personal_best_zkj (np.ndarray): The personal best position of the particle's `z_{kj}`-th dimension cluster center.
    - global_best_zkj (np.ndarray): The global best position of the particle's `z_{kj}`-th dimension cluster center.

    Returns:
    - np.ndarray: The updated position of the particle's `z_{kj}`-th dimension cluster center for the next iteration.

    The function calculates the new position by blending the current position with
    influences from the personal and global best positions, modulated by random
    coefficients, aiming to explore and exploit the search space effectively.
    """
    r3 = np.random.rand()
    r4 = np.random.rand()
    new_zkj = (
        current_zkj
        + r3 * (personal_best_zkj - current_zkj)
        + r4 * (global_best_zkj - current_zkj)
    )

    return new_zkj


def pso_clustering(
    data: np.ndarray, P: int, itermax: int, m: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Description:
    - Perform the Particle Swarm Optimization (PSO) algorithm for clustering
    - Calculate the fitness values of the particles
    - Update the positions of the particles and their cluster centers
    """
    particles = [np.random.rand(m, m) for _ in range(P)]
    clusters, centers = zip(*[custom_kmeans(data, particle) for particle in particles])
    fp = np.array(
        [
            calculate_intra_cluster_fitness(data, cl, ct)
            for cl, ct in zip(clusters, centers)
        ]
    )
    best_particle_positions = [particle.copy() for particle in particles]

    best_particle_fitness = np.zeros((P, itermax))

    t = 0

    with tqdm.tqdm(total=itermax) as pbar:

        while t < itermax:
            best_particle_idx = np.argmin(fp)
            t += 1

            for p in range(P):
                particles[p] = move_particle(
                    particles[p],
                    best_particle_positions[p],
                    best_particle_positions[best_particle_idx],
                )

                k = np.random.randint(0, m)
                particles[p][k] = move_particle_zkj(
                    particles[p][k],
                    best_particle_positions[p][k],
                    best_particle_positions[best_particle_idx][k],
                )

                new_clusters, new_centers = custom_kmeans(data, particles[p])
                new_fp = calculate_adaptive_cluster_fitness(
                    data, clusters[p], centers[p]
                )

                if new_fp < fp[p]:
                    fp[p] = new_fp
                    best_particle_positions[p] = particles[p].copy()
                    clusters = list(clusters)
                    centers = list(centers)

                    clusters[p] = new_clusters
                    centers[p] = new_centers

                best_particle_fitness[p, t - 1] = new_fp

            pbar.update(1)

    return best_particle_fitness, clusters, fp


def main(dataset_names: list[str]) -> None:
    """
    Description:
    - Run the PSO clustering algorithm on the given datasets
    - Calculate the error rates, computation times, and fitness values
    - Plot the results
    """
    logging.info("Running PSO Clustering Simulation")
    error_rates, computation_times = [], []
    best_obj_values, avg_obj_values, worst_obj_values = [], [], []
    actual_clusters, predicted_clusters = [], []

    for dataset_name in dataset_names:
        logging.info(f"Running simulation for {dataset_name} dataset")

        start_time = time.time()

        data, true_labels = load_dataset(dataset_name)
        # Normalize the data
        # scaler = StandardScaler()
        # data = scaler.fit_transform(data)

        n, m = data.shape

        # Values from the paper for particles and iterations
        P = 250
        itermax = m * 15

        best_particle_fitness, clusters, fp = pso_clustering(data, P, itermax, m)

        # Calculate best, average, and worst objective function values
        best_obj_value = np.min(best_particle_fitness)
        avg_obj_value = np.mean(best_particle_fitness)
        worst_obj_value = np.max(best_particle_fitness)

        # Display the results
        logging.info(f"Best Objective Function Value: {best_obj_value:.4f}")
        logging.info(f"Average Objective Function Value: {avg_obj_value:.4f}")
        logging.info(f"Worst Objective Function Value: {worst_obj_value:.4f}")

        best_particle_idx = np.argmin(fp)
        predicted_labels = clusters[best_particle_idx]
        er = error_rate(true_labels, predicted_labels)
        logging.info(f"Error Rate: {er:.2f}%")

        end_time = time.time()
        computation_time = end_time - start_time
        logging.info(f"Computation Time: {computation_time:.2f} seconds")

        unique_true_labels = len(np.unique(true_labels))
        avg_clusters_predicted = np.mean([len(np.unique(cl)) for cl in clusters])
        actual_clusters.append(unique_true_labels)
        predicted_clusters.append(avg_clusters_predicted)
        logging.info(f"Actual number of clusters: {unique_true_labels}")
        logging.info(
            f"Predicted number of clusters (avg): {avg_clusters_predicted:.2f}"
        )

        error_rates.append(er)
        computation_times.append(computation_time)
        best_obj_values.append(best_obj_value)
        avg_obj_values.append(avg_obj_value)
        worst_obj_values.append(worst_obj_value)

        logging.info(f"Generating cluster plot for {dataset_name} dataset...")
        predicted_labels = clusters[best_particle_idx]
        centers = np.array(
            [
                np.mean(data[predicted_labels == k], axis=0)
                for k in np.unique(predicted_labels)
            ]
        )
        plot_clusters(data, true_labels, predicted_labels, centers, dataset_name)

    logging.info("Generating aggregated plots...")
    plot_aggregated_measurements(
        dataset_names,
        error_rates,
        computation_times,
        best_obj_values,
        avg_obj_values,
        worst_obj_values,
        actual_clusters,
        predicted_clusters,
    )

    logging.info("Simulation complete. Data plots saved to 'data' directory.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run PSO clustering algorithm on selected datasets."
    )
    parser.add_argument(
        "-d",
        "--datasets",
        nargs="+",
        default=["iris"],
        help="List of dataset names to process",
    )
    parser.add_argument(
        "-a", "--all", action="store_true", help="Process all available datasets"
    )

    args = parser.parse_args()

    # Validate dataset names
    for dataset_name in args.datasets:
        if dataset_name not in ["iris", "wine", "thyroid", "cmc", "glass"]:
            raise ValueError("Invalid dataset name")

    # Run the simulation for all datasets
    if args.all:
        main(["iris", "wine", "thyroid", "cmc", "glass"])

    main(args.datasets)
