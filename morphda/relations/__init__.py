from morphda.relations.base import (
    MetamorphicRelation,
    RelationResult,
    TransformedCase,
    ViolationWitness,
)
from morphda.relations.universal import UNIVERSAL_RELATIONS
from morphda.relations.filters import FILTER_RELATIONS
from morphda.relations.aggregation import AGGREGATION_RELATIONS
from morphda.relations.grouping import GROUPING_RELATIONS
from morphda.relations.time import TIME_RELATIONS
from morphda.relations.statistics import STATISTICS_RELATIONS
from morphda.relations.joins import JOIN_RELATIONS
from morphda.relations.hardcoding import HARDCODING_RELATIONS

# Full ordered relation set — universal first, then task-conditioned
ALL_RELATIONS: list[MetamorphicRelation] = (
    UNIVERSAL_RELATIONS
    + FILTER_RELATIONS
    + AGGREGATION_RELATIONS
    + GROUPING_RELATIONS
    + TIME_RELATIONS
    + STATISTICS_RELATIONS
    + JOIN_RELATIONS
    + HARDCODING_RELATIONS
)

__all__ = [
    "MetamorphicRelation",
    "RelationResult",
    "TransformedCase",
    "ViolationWitness",
    "UNIVERSAL_RELATIONS",
    "FILTER_RELATIONS",
    "AGGREGATION_RELATIONS",
    "GROUPING_RELATIONS",
    "TIME_RELATIONS",
    "STATISTICS_RELATIONS",
    "JOIN_RELATIONS",
    "HARDCODING_RELATIONS",
    "ALL_RELATIONS",
]
