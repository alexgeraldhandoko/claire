from helper_classes import ConfusionMatrix, PerformanceMetrics

def get_data_from_confusion_matrices(matrices: list[ConfusionMatrix]) -> PerformanceMetrics:
    accuracy_sum = 0
    recall_sum = 0
    precision_sum = 0
    f1_sum = 0

    for matrix in matrices:
        accuracy_sum += matrix.get_accuracy()
        recall_sum += matrix.get_recall()
        precision_sum += matrix.get_precision()
        f1_sum += matrix.get_f1()

    return PerformanceMetrics(
        accuracy = accuracy_sum / len(matrices),
        recall = recall_sum / len(matrices),
        precision = precision_sum / len(matrices),
        f1 = f1_sum / len(matrices)
    )