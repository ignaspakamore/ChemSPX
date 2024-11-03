from multiprocessing import Pool
from itertools import accumulate
import numpy as np
import os


def calc_batch_sizes(n_tasks: int, n_workers: int) -> list:
    """Divide `n_tasks` optimally between n_workers to get batch_sizes.

    Guarantees batch sizes won't differ for more than 1.

    Example:
    # >>>calc_batch_sizes(23, 4)
    # Out: [6, 6, 6, 5]

    In case you're going to use numpy anyway, use np.array_split:
    [len(a) for a in np.array_split(np.arange(23), 4)]
    # Out: [6, 6, 6, 5]
    """
    x = int(n_tasks / n_workers)
    y = n_tasks % n_workers
    batch_sizes = [x + (y > 0)] * y + [x] * (n_workers - y)

    return batch_sizes


def build_batch_ranges(batch_sizes: list) -> list:
    """Build batch_ranges from list of batch_sizes.

    Example:
    # batch_sizes [6, 6, 6, 5]
    # >>>build_batch_ranges(batch_sizes)
    # Out: [range(0, 6), range(6, 12), range(12, 18), range(18, 23)]
    """
    upper_bounds = [*accumulate(batch_sizes)]
    lower_bounds = [0] + upper_bounds[:-1]
    batch_ranges = [range(l, u) for l, u in zip(lower_bounds, upper_bounds)]

    return batch_ranges


def convert_data_to_spheres(data: list, radius: float) -> np.array:
    spheres = np.empty(len(data), dtype="object")
    for i, point in enumerate(data):
        spheres[i] = {"center": point, "radius": radius}
    return spheres


def monte_carlo_worker(batch_range: list) -> int:
    count_inside = 0
    for _ in batch_range:
        point = np.random.uniform(
            boundary_conditions["min"], boundary_conditions["max"]
        )
        for sphere in spheres:
            if np.linalg.norm(point - sphere["center"]) <= sphere["radius"]:
                count_inside += 1
                break
            else:
                continue

    return count_inside


def initializer(data: np.array, r: int, bounds: list) -> None:

    global spheres
    global boundary_conditions
    global radius

    radius = r
    spheres = convert_data_to_spheres(data, radius)
    boundary_conditions = {"min": bounds[0], "max": bounds[1]}


def monte_carlo(
    data: np.array, radius: float, bounds: list, n_workers: int, num_samples: int
) -> np.array:

    batch_sizes = calc_batch_sizes(num_samples, n_workers=n_workers)
    batch_ranges = build_batch_ranges(batch_sizes)

    with Pool(n_workers, initializer, (data, radius, bounds)) as pool:
        results = pool.map(monte_carlo_worker, batch_ranges)

    return np.array(results)


def compute_total_volume(bounds: list) -> float:
    min_bound = np.array(bounds[0])
    max_bound = np.array(bounds[1])

    hyper_rectangle_vol = 1

    for i, j in zip(min_bound, max_bound):
        d = abs(i - j)
        hyper_rectangle_vol *= d

    return hyper_rectangle_vol


def explored_space_ratio(
    data: np.array,
    bounds: list,
    radius: float,
    n_workers: int = os.cpu_count(),
    num_samples: int = 1000000,
) -> dict:

    mc_result = monte_carlo(
        data,
        radius,
        bounds,
        n_workers,
        num_samples,
    )
    mc_result = sum(mc_result)

    # total_volume = compute_total_volume(bounds)

    occupied_ratio = mc_result / num_samples
    free_ratio = 1 - occupied_ratio

    # occupied_vol = ratio * total_volume

    # occupied_ratio = occupied_vol / total_volume
    # free_ratio = 1 - occupied_ratio

    return {"free_space": free_ratio, "occupied_space": occupied_ratio}
