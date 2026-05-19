"""Graph memory layer for AgentKeeper.

Provides a relational structure on top of the cognitive state.
Facts capture prose ("Alice works at Acme"), triples capture structure
(Alice -[works_at]-> Acme). Use facts for recall, triples for traversal.

Triples are stored alongside facts on the CSO and obey the same
retention/GDPR rules (TTL, protected flag, gdpr_export, gdpr_purge).

Public surface:

- `Triple`         : a (subject, predicate, object) directed relation
- `RelationGraph`  : a view over a CSO's triples with traversal queries
"""

from .relation_graph import RelationGraph
from .triple import Triple

__all__ = ["RelationGraph", "Triple"]
