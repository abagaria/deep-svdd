import pdb
from itertools import chain
import numpy as np
import matplotlib.pyplot as plt


def plot_test_predictions(predictions):
    positive_points = np.array([prediction[0].cpu().numpy().squeeze(0) for prediction in predictions if prediction[1].item() == 1])
    negative_points = np.array([prediction[0].cpu().numpy().squeeze(0) for prediction in predictions if prediction[1].item() == 0])
    plt.scatter(positive_points[:, 0], positive_points[:, 1], label="inliers")
    plt.scatter(negative_points[:, 0], negative_points[:, 1], label="outliers")
    plt.legend()
    plt.show()


def get_true_positives(predictions):
    p = []
    for batch_predictions in predictions:
        points = batch_predictions[0]
        labels = batch_predictions[1]
        for point, label in zip(points, labels):
            if label.item() == 1:
                p.append(point.cpu().numpy())
    return np.array(p)


def get_false_positives(predictions):
    p = []
    for batch_predictions in predictions:
        points = batch_predictions[0]
        labels = batch_predictions[1]
        for point, label in zip(points, labels):
            if label.item() == 0:
                p.append(point.cpu().numpy())
    return np.array(p)


def get_true_negatives(predictions):
    p = []
    for batch_predictions in predictions:
        points = batch_predictions[0]
        labels = batch_predictions[1]
        for point, label in zip(points, labels):
            if label.item() == 0:
                p.append(point.cpu().numpy())
    return np.array(p)


def get_false_negatives(predictions):
    p = []
    for batch_predictions in predictions:
        points = batch_predictions[0]
        labels = batch_predictions[1]
        for point, label in zip(points, labels):
            if label.item() == 1:
                p.append(point.cpu().numpy())
    return np.array(p)


def plot_predictions(inlier_predictions, outlier_predictions):
    """
    Each "prediction" is of the form data, label tuple
    :param inlier_predictions: Predictions on training data (should be inliers)
    :param outlier_predictions: Predictions on test data (should be outliers)
    """
    correct_inlier_predictions = get_true_positives(inlier_predictions)
    wrong_inlier_predictions = get_false_positives(inlier_predictions)

    correct_outlier_predictions = get_true_negatives(outlier_predictions)
    wrong_outlier_predictions = get_false_negatives(outlier_predictions)

    plt.scatter(correct_inlier_predictions[:, 0], correct_inlier_predictions[:, 1], label="true inliers")
    plt.scatter(wrong_inlier_predictions[:, 0], wrong_inlier_predictions[:, 1], label="false inliers")
    plt.scatter(correct_outlier_predictions[:, 0], correct_outlier_predictions[:, 1], label="true outliers")
    plt.scatter(wrong_outlier_predictions[:, 0], wrong_outlier_predictions[:, 1], label="false outliers")
    plt.legend()
    plt.show()


def get_inlier_percentage(predictions):
    num_inliers = 0
    num_labels = 0
    for batch_predictions in predictions:
        labels = batch_predictions[1]
        num_inliers += labels.sum().item()
        num_labels += labels.shape[0]
    return float(num_inliers) / num_labels
