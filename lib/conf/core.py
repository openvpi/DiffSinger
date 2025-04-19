from contextvars import ContextVar
from typing import Any, Optional, ClassVar

from pydantic import BaseModel, Field, model_validator

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


if __name__ == '__main__':
    # Example usage
    # Scope definitions (bitmask flags)
    SCOPE_A = 0x1  # 00000001
    SCOPE_B = 0x2  # 00000010
    SCOPE_C = 0x4  # 00000100
    SCOPE_D = 0x8  # 00001000
    # Additional scopes can be defined as powers of 2

    class DatabaseConfig(ConfigBaseModel):
        host: str

        # Field only available in SCOPE_A
        replica_set: Optional[str] = Field(
            default=None,
            json_schema_extra={"scope": SCOPE_A}
        )

        # Field available in both SCOPE_A and SCOPE_B
        timeout: Optional[int] = Field(
            default=30,
            json_schema_extra={"scope": SCOPE_A | SCOPE_B}
        )

        # Nested config only for SCOPE_B
        sharding: Optional["ShardingConfig"] = Field(
            default=None,
            json_schema_extra={"scope": SCOPE_B}
        )


    class ShardingConfig(ConfigBaseModel):
        shard_key: str
        clusters: int = Field(
            default=3,
            json_schema_extra={"scope": SCOPE_B | SCOPE_C}
        )


    # Creating configuration for SCOPE_A
    config_a = DatabaseConfig.model_validate({
        "host": "primary.db.example",
        "replica_set": "rs0",
        "timeout": 60,
        "sharding": {  # This field will be ignored for SCOPE_A
            "shard_key": "_id",
            "clusters": 5
        }
    }, scope=SCOPE_A)
    print(config_a.model_dump())
    # Output: {'scope': 1, 'host': 'primary.db.example', 'replica_set': 'rs0', 'timeout': 60}

    # Creating configuration for SCOPE_B
    config_b = DatabaseConfig.model_validate({
        "host": "cluster.db.example",
        "replica_set": "ignored",  # Will be filtered out
        "timeout": 45,
        "sharding": {
            "shard_key": "user_id",
            "clusters": 5
        }
    }, scope=SCOPE_B)
    print(config_b.model_dump())
    # Output: {'scope': 2, 'host': 'cluster.db.example', 'timeout': 45,
    #          'sharding': {'scope': 2, 'shard_key': 'user_id', 'clusters': 5}}
