import numpy as np


def score(points: list[tuple], curr_point: tuple, center: tuple, mu=0, sigma=0.4) -> float:
    return distance_score(curr_point, center, mu, sigma) * stability_score(points)


def distance_score(point: tuple, center: tuple, mu=0, sigma=0.4) -> float:
    # Calculate the distance between the point and the center of the key
    x, y = point
    x_center, y_center = center

    d = np.sqrt((x - x_center)**2 + (y - y_center)**2)

    denominator = sigma * np.sqrt(2 * np.pi)
    exponent = -((d - mu)**2) / (2 * sigma**2)

    return np.exp(exponent) / denominator

def stability_score(points: list[tuple]) -> float:
    # I received a list with the last 20 points
    # We are working with 60Hz, so 1/60 of a second is the time between each point

    dt = 1/60
    total = 0

    for i in range(1, len(points)):
        total += np.sqrt((points[i][0] - points[i-1][0])**2 + (points[i][1] - points[i-1][1])**2) / dt

    return len(points)/total if total != 0 else 0