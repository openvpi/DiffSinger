import abc
import operator as _operator
import re

from pydantic import BaseModel

KEY_PATH_HEAD = re.compile(r"[^.[]*")
KEY_PATH_OTHER = re.compile(r"\.([^.[]*)|\[(.*?)]")


# This function is adapted from omegaconf._utils
def split_path(key: str) -> list[str]:
    """
    Split a full key path into its individual components.

    This is similar to `key.split(".")` but also works with the getitem syntax:
        "a.b"       -> ["a", "b"]
        "a[b]"      -> ["a", "b"]
        "a.b[c].d" -> ["a", "b", "c", "d"]
    """
    # Obtain the first part of the key (in docstring examples: a, a, a)
    first = KEY_PATH_HEAD.match(key)
    assert first is not None
    first_stop = first.span()[1]

    # `tokens` will contain all elements composing the key.
    tokens = [key[0:first_stop]]
    assert len(tokens[0]) > 0, f"Empty key component found in {key}"

    # Optimization in case `key` has no other component: we are done.
    if first_stop == len(key):
        return tokens

    # Identify other key elements (in docstring examples: b, b, b/c/d)
    others = KEY_PATH_OTHER.findall(key[first_stop:])

    # There are two groups in the `KEY_PATH_OTHER` regex: one for keys starting
    # with a dot (.b, .d) and one for keys starting with a bracket ([b], [c]).
    # Only one group can be non-empty.
    tokens += [dot_key if dot_key else bracket_key for dot_key, bracket_key in others]
    assert all(len(token) > 0 for token in tokens[1:]), f"Empty key component found in {key}"

    return tokens


class ConfigOperationContext:
    def __init__(self, root: BaseModel, current_path: list[str | int], current_value, scope: int):
        self.root = root
        self.current_path = current_path
        self.current_value = current_value
        self.scope = scope


class ConfigOperationBase(abc.ABC):
    @abc.abstractmethod
    def resolve(self, context: ConfigOperationContext):
        pass

    def __add__(self, other):
        return BinaryOperation(self, other, "+")

    def __sub__(self, other):
        return BinaryOperation(self, other, "-")

    def __mul__(self, other):
        return BinaryOperation(self, other, "*")

    def __truediv__(self, other):
        return BinaryOperation(self, other, "/")

    def __floordiv__(self, other):
        return BinaryOperation(self, other, "//")

    def __mod__(self, other):
        return BinaryOperation(self, other, "%")

    def __pow__(self, other):
        return BinaryOperation(self, other, "**")

    def __eq__(self, other):
        return BinaryOperation(self, other, "==")

    def __ne__(self, other):
        return BinaryOperation(self, other, "!=")

    def __lt__(self, other):
        return BinaryOperation(self, other, "<")

    def __le__(self, other):
        return BinaryOperation(self, other, "<=")

    def __gt__(self, other):
        return BinaryOperation(self, other, ">")

    def __ge__(self, other):
        return BinaryOperation(self, other, ">=")

    def __and__(self, other):
        return BinaryOperation(self, other, "&")

    def __or__(self, other):
        return BinaryOperation(self, other, "|")

    def __xor__(self, other):
        return BinaryOperation(self, other, "^")

    def __neg__(self):
        return UnaryOperation(self, "-")

    def __invert__(self):
        return UnaryOperation(self, "~")

    def __abs__(self):
        return UnaryOperation(self, "abs")

    def __round__(self):
        return UnaryOperation(self, "round")


class This(ConfigOperationBase):
    def resolve(self, context: ConfigOperationContext):
        return context.current_value


class Reference(ConfigOperationBase):
    def __init__(self, path: str):
        self.components = split_path(path)

    def resolve(self, context: ConfigOperationContext):
        current = context.root
        for component in self.components:
            if isinstance(current, (tuple, list)):
                if not component.isdigit():
                    raise ValueError(f"Invalid index '{component}' for list or tuple.")
                index = int(component)
                if index < 0:
                    index += len(current)
                if index < 0 or index >= len(current):
                    return None
            else:
                current = getattr(current, component, None)
            if current is None:
                return None
        return current


class GetContext(ConfigOperationBase):
    def __init__(self, attr: str = None):
        self.attr = attr

    def resolve(self, context: ConfigOperationContext):
        if self.attr is None:
            return context
        return getattr(context, self.attr, None)


_BINARY_OPS = {
    "+": _operator.add,
    "-": _operator.sub,
    "*": _operator.mul,
    "/": _operator.truediv,
    "%": _operator.mod,
    "**": _operator.pow,
    "//": _operator.floordiv,
    "==": _operator.eq,
    "!=": _operator.ne,
    "<": _operator.lt,
    "<=": _operator.le,
    ">": _operator.gt,
    ">=": _operator.ge,
    "&": _operator.and_,
    "|": _operator.or_,
    "^": _operator.xor,
    "and": lambda a, b: a and b,
    "or": lambda a, b: a or b,
    "in": lambda a, b: a in b,
}

_UNARY_OPS = {
    "-": _operator.neg,
    "~": _operator.invert,
    "abs": abs,
    "round": round,
    "not": _operator.not_,
    "exists": lambda x: x is not None,
    "missing": lambda x: x is None,
}

_AGGREGATION_OPS = {
    "min": lambda values: min(*values),
    "max": lambda values: max(*values),
    "len": lambda values: len(*values),
    "sum": lambda values: sum(*values),
    "avg": lambda values: sum(*values) / len(*values),
    "all": lambda values: all(*values),
    "any": lambda values: any(*values),
    "list": lambda values: list(*values),
    "set": lambda values: set(*values),
}


class BinaryOperation(ConfigOperationBase):
    def __init__(self, left, right, op):
        self.left = left
        self.right = right
        self.op = op

    def resolve(self, context: ConfigOperationContext):
        left_value = self.left.resolve(context) if isinstance(self.left, ConfigOperationBase) else self.left
        right_value = self.right.resolve(context) if isinstance(self.right, ConfigOperationBase) else self.right
        if self.op not in _BINARY_OPS:
            raise ValueError(f"Unsupported operator: {self.op}")
        return _BINARY_OPS[self.op](left_value, right_value)


class UnaryOperation(ConfigOperationBase):
    def __init__(self, operand, op):
        self.operand = operand
        self.op = op

    def resolve(self, context: ConfigOperationContext):
        operand_value = (
            self.operand.resolve(context) if isinstance(self.operand, ConfigOperationBase) else self.operand
        )
        if self.op not in _UNARY_OPS:
            raise ValueError(f"Unsupported operator: {self.op}")
        return _UNARY_OPS[self.op](operand_value)


class Aggregation(ConfigOperationBase):
    def __init__(self, *args, op, key=None):
        self.args = args
        self.op = op
        self.key = key

    def resolve(self, context: ConfigOperationContext):
        if isinstance(self.args, ConfigOperationBase):
            args = self.args.resolve(context)
        else:
            args = self.args
        values = []
        for arg in args:
            if isinstance(arg, ConfigOperationBase):
                value = arg.resolve(context)
            else:
                value = arg
            if self.key:
                value = self.key(value)
            values.append(value)
        if self.op not in _AGGREGATION_OPS:
            raise ValueError(f"Unsupported aggregation operation: {self.op}")
        return _AGGREGATION_OPS[self.op](values)


class Coalesce(ConfigOperationBase):
    def __init__(self, *args):
        self.args = args

    def resolve(self, context: ConfigOperationContext):
        for arg in self.args:
            if isinstance(arg, ConfigOperationBase):
                value = arg.resolve(context)
            else:
                value = arg
            if value is not None:
                return value
        return None


class MapOperation(ConfigOperationBase):
    def __init__(self, iterable, f):
        self.iterable = iterable
        self.func = f

    def resolve(self, context: ConfigOperationContext):
        if isinstance(self.iterable, ConfigOperationBase):
            iterable = self.iterable.resolve(context)
        else:
            iterable = self.iterable
        res = []
        for item in iterable:
            if isinstance(item, ConfigOperationBase):
                item = item.resolve(context)
            res.append(self.func(item))
        return res


class FilterOperation(ConfigOperationBase):
    def __init__(self, iterable, f):
        self.iterable = iterable
        self.func = f

    def resolve(self, context: ConfigOperationContext):
        if isinstance(self.iterable, ConfigOperationBase):
            iterable = self.iterable.resolve(context)
        else:
            iterable = self.iterable
        res = []
        for item in iterable:
            if isinstance(item, ConfigOperationBase):
                item = item.resolve(context)
            if self.func(item):
                res.append(item)
        return res


class IfElse(ConfigOperationBase):
    def __init__(self, condition, if_true, if_false):
        self.condition = condition
        self.if_true = if_true
        self.if_false = if_false

    def resolve(self, context: ConfigOperationContext):
        if isinstance(self.condition, ConfigOperationBase):
            condition = self.condition.resolve(context)
        else:
            condition = self.condition
        if condition:
            branch = self.if_true
        else:
            branch = self.if_false
        if isinstance(branch, ConfigOperationBase):
            branch = branch.resolve(context)
        return branch


class Function(ConfigOperationBase):
    def __init__(self, f, *args):
        self.func = f
        self.args = args

    def resolve(self, context: ConfigOperationContext):
        resolved_args = [arg.resolve(context) if isinstance(arg, ConfigOperationBase) else arg for arg in self.args]
        return self.func(*resolved_args)


def this():
    """
    Create a reference to the current field.
    """
    return This()


def ref(path: str):
    """
    Create a reference to a field from the root of the config.
    :param path: the path to the field, e.g. "a.b.c" or "a[0].b".
    """
    return Reference(path)


def ctx(attr: str = None):
    """
    Create a reference to the context of the current field.
    :param attr: the attribute of the context to reference; if None, the entire context will be returned.
    """
    return GetContext(attr)


def abs_(operand):
    """
    Create an absolute value operation.
    """
    return UnaryOperation(operand, "abs")


def round_(operand):
    """
    Create a round operation.
    """
    return UnaryOperation(operand, "round")


def in_(left, right):
    """
    Create a membership test operation.
    :param left: the value to check.
    :param right: the iterable to check against.
    """
    return BinaryOperation(left, right, "in")


def and_(arg1, arg2):
    """
    Create a logical AND operation.
    """
    return BinaryOperation(arg1, arg2, "and")


def or_(arg1, arg2):
    """
    Create a logical OR operation.
    """
    return BinaryOperation(arg1, arg2, "or")


def not_(operand):
    """
    Create a logical NOT operation.
    """
    return UnaryOperation(operand, "not")


def exists(operand):
    """
    Create an operation that checks if the operand is not None.
    """
    return UnaryOperation(operand, "exists")


def missing(operand):
    """
    Create an operation that checks if the operand is None.
    """
    return UnaryOperation(operand, "missing")


def if_(condition, true_value, false_value):
    """
    Create an if-else operation.
    """
    return IfElse(condition, true_value, false_value)


def min_(*args, key=None):
    """
    Create an operation to get the minimum from an iterable or multiple values.
    :param args: The values to aggregate.
    :param key: A function to apply to each element before aggregation (optional).
    """
    return Aggregation(*args, op="min", key=key)


def max_(*args, key=None):
    """
    Create an operation to get the maximum from an iterable or multiple values.
    :param args: The values to aggregate.
    :param key: A function to apply to each element before aggregation (optional).
    """
    return Aggregation(*args, op="max", key=key)


def len_(iterable):
    """
    Create an operation to get the length of an iterable.
    """
    return Aggregation(iterable, op="len")


def sum_(iterable, key=None):
    """
    Create an operation to get the sum from an iterable.
    :param iterable: The values to aggregate.
    :param key: A function to apply to each element before aggregation (optional).
    """
    return Aggregation(iterable, op="sum", key=key)


def avg(iterable, key=None):
    """
    Create an operation to get the average from an iterable.
    :param iterable: The values to aggregate.
    :param key: A function to apply to each element before aggregation (optional).
    """
    return Aggregation(iterable, op="avg", key=key)


def all_(iterable, key=None):
    """
    Create an operation to check if all elements in an iterable are True.
    :param iterable: The values to aggregate.
    :param key: A function to apply to each element before aggregation (optional).
    """
    return Aggregation(iterable, op="all", key=key)


def any_(iterable, key=None):
    """
    Create an operation to check if any element in an iterable is True.
    :param iterable: The values to aggregate.
    :param key: A function to apply to each element before aggregation (optional).
    """
    return Aggregation(iterable, op="any", key=key)


def list_(*args):
    """
    Create a list formation operation.
    """
    return Aggregation(*args, op="list")


def set_(*args):
    """
    Create a set formation operation.
    """
    return Aggregation(*args, op="set")


def coalesce(*args):
    """
    Create a coalesce operation that returns the first non-None value.
    """
    return Coalesce(*args)


def map_(iterable, f):
    """
    Create a map operation.
    :param iterable: the iterable to map over.
    :param f: the function to apply to each element.
    """
    return MapOperation(iterable, f)


def filter_(iterable, f):
    """
    Create a filter operation.
    :param iterable: the iterable to filter.
    :param f: the function to apply to each element.
    """
    return FilterOperation(iterable, f)


def func(f: callable, *args):
    """
    Create a custom function operation.
    :param f: the function to call.
    :param args: the arguments to pass to the function.
    """
    return Function(f, *args)
