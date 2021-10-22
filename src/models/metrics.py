class RegressionMetrics:
    def __init__(self, rmsle: float, mape: float):
        self.rmsle = rmsle
        self.mape = mape

    def to_dict(self):
        return {
            'rmsle': self.rmsle,
            'mape': self.mape
        }


class RegressionEvaluation:
    def __init__(self, model: str, train_metrics: RegressionMetrics, test_metrics: RegressionMetrics):
        self.model = model
        self.train_metrics = train_metrics
        self.test_metrics = test_metrics

    def to_dict(self):
        return {
            'model_name': self.model,
            'train_mape': self.train_metrics.mape,
            'train_rmsle': self.train_metrics.rmsle,
            'test_mape': self.test_metrics.mape,
            'test_rmsle': self.test_metrics.rmsle
        }
