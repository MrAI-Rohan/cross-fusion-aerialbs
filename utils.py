class FilterByBR:
    """Filter patches by ratio of no. of building pixels in them."""
    def __init__(self, building_threshold,):
        self.building_threshold = building_threshold

    def __call__(self, mask):
        return (mask > 0).sum() / mask.size > self.building_threshold
    

def compute_metrics(tp, fp, fn, tn):
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    accuracy = (tp + tn) / (tp + fp + fn + tn + 1e-6)
    pos_iou = tp / (tp + fp + fn + 1e-6)
    neg_iou = tn / (tn + fp + fn + 1e-6)
    mean_iou = (pos_iou + neg_iou) / 2

    return {
        "pos_iou": pos_iou,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "neg_iou": neg_iou,
        "mean_iou": mean_iou,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn
    }