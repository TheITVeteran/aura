################################################################################

"""tests/test_federated_query.py
Unit test for Federated Query logic in BeliefGraph.
"""
import unittest

from core.world_model.belief_graph import BeliefGraph
from core.container import ServiceContainer


class BeliefSyncProbe:
    def __init__(self):
        self.remote_beliefs = []
        self.queries = []

    async def query_peers(self, query):
        self.queries.append(query)
        return [dict(belief) for belief in self.remote_beliefs]


class TestFederatedQuery(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        ServiceContainer.clear()

        self.graph = BeliefGraph(persist_path=":memory:")
        self.graph.update_belief("User", "is", "Happy", 0.9)

        self.sync_service = BeliefSyncProbe()
        ServiceContainer.register_instance("belief_sync", self.sync_service)

    async def test_merged_query(self):
        self.sync_service.remote_beliefs = [
            {"source": "User", "relation": "likes", "target": "Coffee", "confidence": 0.9}
        ]

        results = await self.graph.query_federated("User")

        sources = [b["source"] for b in results]
        targets = [b["target"] for b in results]

        self.assertIn("User", sources)
        self.assertIn("Happy", targets)
        self.assertIn("Coffee", targets)

        coffee_belief = next(b for b in results if b["target"] == "Coffee")
        self.assertAlmostEqual(coffee_belief["confidence"], 0.72)
        self.assertEqual(self.sync_service.queries, ["User"])

if __name__ == '__main__':
    unittest.main()


##
