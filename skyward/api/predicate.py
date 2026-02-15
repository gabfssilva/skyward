from typing import Literal

type Comparison = Literal['==', '!=', '>=', '<=', '!=', 'in', 'not in', 'any', 'none']

class Match[T]:
    value: T
    comparison: Comparison

type Predicate[T] = T | Match[T]
