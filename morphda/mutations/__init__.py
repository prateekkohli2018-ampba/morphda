from morphda.mutations.base import MutationOperator, MutantRecord
from morphda.mutations.filters import FILTER_OPERATORS
from morphda.mutations.aggregation import AGGREGATION_OPERATORS
from morphda.mutations.grouping import GROUPING_OPERATORS
from morphda.mutations.hardcoding import HARDCODING_OPERATORS
from morphda.mutations.joins import JOIN_OPERATORS

ALL_OPERATORS: list[MutationOperator] = (
    FILTER_OPERATORS
    + AGGREGATION_OPERATORS
    + GROUPING_OPERATORS
    + HARDCODING_OPERATORS
    + JOIN_OPERATORS
)

__all__ = [
    "MutationOperator", "MutantRecord",
    "ALL_OPERATORS",
    "FILTER_OPERATORS",
    "AGGREGATION_OPERATORS",
    "GROUPING_OPERATORS",
    "HARDCODING_OPERATORS",
    "JOIN_OPERATORS",
]
