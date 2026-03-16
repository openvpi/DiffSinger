from contextvars import ContextVar
from typing import Any, ClassVar

from pydantic import BaseModel, model_validator

from .ops import ConfigOperationContext, ConfigOperationBase, split_path

__all__ = [
    "ConfigBaseModel",
]

# Context variable holds current scope
_current_scope: ContextVar[int] = ContextVar("_current_scope")


class ConfigBaseModel(BaseModel):
    """
    Base model for scope-based configuration system.
    Inherit from this class to create scoped configuration models.

    Key Features:
    - Fields can be marked with scope flags using Field(json_schema_extra={"scope": SCOPE_A|SCOPE_B})
    - Only fields matching the current scope will be validated and included in outputs
    - Scope context automatically propagates to nested models
    """

    __field_scopes__: ClassVar[dict] = {}  # {field_name: scope_bitmask}

    def __init__(self, **data):
        token = _current_scope.set(0)
        try:
            super().__init__(**data)
        finally:
            _current_scope.reset(token)

    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs):
        """
        Collect scope markers from all fields during subclass creation.
        """
        cls.__field_scopes__ = {}
        for name, field in cls.model_fields.items():
            if field.json_schema_extra and "scope" in field.json_schema_extra:
                cls.__field_scopes__[name] = field.json_schema_extra["scope"]

    @classmethod
    def model_validate(cls, obj: Any, scope: int = 0, **kwargs):
        """
        Entry point for model validation.
        Sets the context scope before any validation occurs.
        """
        # Set context for the entire validation chain
        token = _current_scope.set(scope)
        try:
            return super().model_validate(obj, **kwargs)
        finally:
            _current_scope.reset(token)

    @model_validator(mode="after")
    def validate_fields_by_scope(self) -> "ConfigBaseModel":
        """
        Post-validation: Remove fields not matching current scope.
        """
        current_scope = _current_scope.get()
        for name in type(self).model_fields:
            if name in self.__field_scopes__ and not (current_scope & self.__field_scopes__[name]):
                delattr(self, name)
        return self

    def _resolve_recursive(self, current: "ConfigBaseModel", context: ConfigOperationContext):
        """
        Recursively resolve all dynamic expressions in the config.
        :param current: The current model instance.
        :param context: The context for resolving dynamic expressions.
        """
        for field_name, field_info in type(current).model_fields.items():
            field_scope = current.__field_scopes__.get(field_name)
            if field_scope is not None and not field_scope & context.scope:
                continue
            context.current_path.append(field_name)
            value = getattr(current, field_name)
            if isinstance(value, ConfigBaseModel):
                self._resolve_recursive(value, context)
            elif isinstance(value, list):
                for i, item in enumerate(value):
                    context.current_path.append(i)
                    if isinstance(item, ConfigBaseModel):
                        self._resolve_recursive(item, context)
                    context.current_path.pop()
            if field_info.json_schema_extra is not None:
                expr = field_info.json_schema_extra.get('dynamic_expr')
                if expr:
                    if isinstance(expr, ConfigOperationBase):
                        context.current_value = value
                        expr = expr.resolve(context)
                    setattr(current, field_name, expr)
            context.current_path.pop()

    def _check_recursive(self, current: "ConfigBaseModel", context: ConfigOperationContext):
        """
        Recursively check all dynamic expressions in the config.
        :param current: The current model instance.
        :param context: The context for checking dynamic expressions.
        """
        for field_name, field_info in type(current).model_fields.items():
            field_scope = current.__field_scopes__.get(field_name)
            if field_scope is not None and not field_scope & context.scope:
                continue
            context.current_path.append(field_name)
            value = getattr(current, field_name)
            if isinstance(value, ConfigBaseModel):
                self._check_recursive(value, context)
            elif isinstance(value, list):
                for i, item in enumerate(value):
                    context.current_path.append(i)
                    if isinstance(item, ConfigBaseModel):
                        self._check_recursive(item, context)
                    context.current_path.pop()
            if field_info.json_schema_extra is not None:
                check = field_info.json_schema_extra.get('dynamic_check')
                if check:
                    context.current_value = value
                    check.run(context)
            context.current_path.pop()

    def _process_nested(self, f, scope: int = 0, path: str = None):
        """
        Process the nested config object under a given path.
        :param f: The function to apply to the object.
        :param scope: The scope mask.
        :param path: Path of the object. None points to the root.
        """
        current = self
        if path:
            parts = split_path(path)
            for p in parts:
                if isinstance(current, (tuple, list)):
                    current = current[int(p)]
                elif isinstance(current, dict):
                    current = current[p]
                else:
                    current = getattr(current, p)
        else:
            parts = []
        if not isinstance(current, ConfigBaseModel):
            return
        context = ConfigOperationContext(
            root=self,
            current_path=parts,
            current_value=current,
            scope=scope
        )
        f(current, context)

    def resolve(self, scope_mask: int = 0, from_path: str = None):
        """
        Resolve all dynamic expressions from a given path in the config.
        :param scope_mask: The scope mask to use for dynamic resolving.
        :param from_path: The path to resolve from. If None, resolve from the root.
        """
        self._process_nested(self._resolve_recursive, scope_mask, from_path)

    def check(self, scope_mask: int = 0, from_path: str = None):
        """
        Check all dynamic expressions from a given path in the config.
        :param scope_mask: The scope mask to use for dynamic checking.
        :param from_path: The path to check from. If None, check from the root.
        """
        self._process_nested(self._check_recursive, scope_mask, from_path)
