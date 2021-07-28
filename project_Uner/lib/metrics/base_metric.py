"""From PyTorch-Ignite."""
import itertools
from abc import ABCMeta, abstractmethod


class NotComputableError(RuntimeError):
    pass


class BaseMetric(metaclass=ABCMeta):
    def __init__(self):
        self.reset()

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def update(self, *args, **kwargs):
        pass

    @abstractmethod
    def compute(self):
        pass

    def __add__(self, other):
        return MetricsLambda(lambda x, y: x + y, self, other)

    def __radd__(self, other):
        return MetricsLambda(lambda x, y: x + y, other, self)

    def __sub__(self, other):
        return MetricsLambda(lambda x, y: x - y, self, other)

    def __rsub__(self, other):
        return MetricsLambda(lambda x, y: x - y, other, self)

    def __mul__(self, other):
        return MetricsLambda(lambda x, y: x * y, self, other)

    def __rmul__(self, other):
        return MetricsLambda(lambda x, y: x * y, other, self)

    def __pow__(self, other):
        return MetricsLambda(lambda x, y: x ** y, self, other)

    def __rpow__(self, other):
        return MetricsLambda(lambda x, y: x ** y, other, self)

    def __mod__(self, other):
        return MetricsLambda(lambda x, y: x % y, self, other)

    def __div__(self, other):
        return MetricsLambda(lambda x, y: x.__div__(y), self, other)

    def __rdiv__(self, other):
        return MetricsLambda(lambda x, y: x.__div__(y), other, self)

    def __truediv__(self, other):
        return MetricsLambda(lambda x, y: x.__truediv__(y), self, other)

    def __rtruediv__(self, other):
        return MetricsLambda(lambda x, y: x.__truediv__(y), other, self)

    def __floordiv__(self, other):
        return MetricsLambda(lambda x, y: x // y, self, other)

    def __getattr__(self, attr):
        def fn(x, *args, **kwargs):
            return getattr(x, attr)(*args, **kwargs)

        def wrapper(*args, **kwargs):
            return MetricsLambda(fn, self, *args, **kwargs)

        return wrapper

    def __getitem__(self, index):
        return MetricsLambda(lambda x: x[index], self)

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, d):
        self.__dict__.update(d)


class MetricsLambda(BaseMetric):
    """From PyTorch-Ignite."""

    def __init__(self, f, *args, **kwargs):
        self.function = f
        self.args = args
        self.kwargs = kwargs
        super().__init__()

    def reset(self):
        for i in itertools.chain(self.args, self.kwargs.values()):
            if isinstance(i, BaseMetric):
                i.reset()

    def update(self, *args, **kwargs):
        pass

    def compute(self):
        materialized = [i.compute() if isinstance(i, BaseMetric) else i for i in self.args]
        materialized_kwargs = {k: (v.compute() if isinstance(v, BaseMetric) else v) for k, v in self.kwargs.items()}
        return self.function(*materialized, **materialized_kwargs)
