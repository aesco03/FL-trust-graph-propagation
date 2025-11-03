"""
Unit tests for TrustGraphStrategy.

Covers basic initialization and aggregate_fit behavior with mock client data.
"""

from tests.common import Mock, pytest, generate_mock_client_data

from src.simulation_strategies.trust_graph_strategy import TrustGraphStrategy


class DummyHistory:
    def insert_single_client_history_entry(self, **kwargs):
        pass

    def insert_round_history_entry(self, **kwargs):
        pass

    def update_client_participation(self, **kwargs):
        pass


class TestTrustGraphStrategy:
    @pytest.fixture
    def strategy(self):
        return TrustGraphStrategy(
            remove_clients=True,
            begin_removing_from_round=2,
            alpha=0.8,
            K=5,
            tau=0.2,
            edge_rule="similarity",
            neighbor_cap=3,
            graph_static=True,
            convergence_eps=1e-3,
            strategy_history=DummyHistory(),
        )

    def test_basic_initialization(self, strategy):
        assert strategy is not None
        assert hasattr(strategy, "alpha")
        assert hasattr(strategy, "K")
        assert hasattr(strategy, "tau")
        assert hasattr(strategy, "neighbor_cap")
        assert hasattr(strategy, "graph_static")
        assert hasattr(strategy, "convergence_eps")

    def test_aggregate_fit_updates_trust(self, strategy):
        mock_results = generate_mock_client_data(num_clients=5)
        aggregated_params, metrics = strategy.aggregate_fit(1, mock_results, [])

        assert aggregated_params is not None
        assert isinstance(metrics, dict)
        assert len(strategy.trust_scores) == 5
        assert len(strategy.self_scores) == 5

    def test_configure_fit_returns_clients(self, strategy):
        # Prepare some trust scores
        strategy.trust_scores = {str(i): 0.5 + i * 0.01 for i in range(5)}
        strategy.current_round = 3

        # Mock client manager .all()
        clients = {str(i): Mock() for i in range(5)}
        for i, c in clients.items():
            c.cid = i

        client_manager = Mock()
        client_manager.all.return_value = clients

        fit_ins_list = strategy.configure_fit(2, Mock(), client_manager)

        assert isinstance(fit_ins_list, list)
        assert len(fit_ins_list) == 5
