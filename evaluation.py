from sklearn import metrics


class Evaluation:

    def __init__(self, y_true, y_pred):
        self.y_true, self.y_pred = y_true, y_pred

        self.tn, self.fp, self.fn, self.tp = metrics.confusion_matrix(y_true.ravel(), y_pred.ravel()).ravel()




