from sklearn.metrics import silhouette_score

def calculate_silhouette_score(data, labels):
    """
    Calculate the silhouette score for a clustering result.

    Parameters:
        data (array-like): The input data.
        labels (array-like): The predicted labels for each data point.

    Returns:
        float: The silhouette score.
    """
    return silhouette_score(data, labels)