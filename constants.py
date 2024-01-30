from dataclasses import dataclass


@dataclass()
class Hyperparameters:
    NUM_TREES = "n_estimators"
    GAMMA = "gamma"
    LEARNING_RATE = "learning_rate"


@dataclass()
class TargetFeature:
    TARGET = "TARGET"


@dataclass()
class Metrics:
    TRAIN_AUC = "Train AUC"
    TRAIN_FSCORE = "Train F1-score"
    TRAIN_PRECISION = "Train Precision"
    TRAIN_RECALL = "Train Recall"
    TEST_AUC = "Test AUC"
    TEST_FSCORE = "Test F1-score"
    TEST_PRECISION = "Test Precision"
    TEST_RECALL = "Test Recall"
    AUC_DIFFERENCE = "AUC Difference"
    FSCORE_DIFFERENCE = 'F1-Score Difference '
    PRECISION_DIFFERENCE = 'Precision Difference'
    RECALL_DIFFERENCE = 'Recall Difference'


HYPERPARAMETER_LIST = [
    Hyperparameters.NUM_TREES,
    Hyperparameters.LEARNING_RATE,
    Hyperparameters.GAMMA,
]

METRIC_LIST = [
    Metrics.TRAIN_AUC,
    Metrics.TRAIN_FSCORE,
    Metrics.TRAIN_PRECISION,
    Metrics.TRAIN_RECALL,
    Metrics.TEST_AUC,
    Metrics.TEST_FSCORE,
    Metrics.TEST_PRECISION,
    Metrics.TEST_RECALL
]

METRIC_DIFFERENCE_LIST = [
    Metrics.AUC_DIFFERENCE,
    Metrics.FSCORE_DIFFERENCE,
    Metrics.PRECISION_DIFFERENCE,
    Metrics.RECALL_DIFFERENCE
]