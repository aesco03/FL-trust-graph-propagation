import numpy as np
import flwr as fl
import logging

from typing import Dict, List, Optional, Tuple, Union

 # Optional: will fallback if sklearn is unavailable

from flwr.common import FitRes, Parameters, Scalar
from flwr.common import EvaluateRes
from flwr.server.strategy.aggregate import weighted_loss_avg
from flwr.common import parameters_to_ndarrays, ndarrays_to_parameters
from flwr.server.client_proxy import ClientProxy

from src.data_models.simulation_strategy_history import SimulationStrategyHistory


class TrustGraphStrategy(fl.server.strategy.FedAvg):
    def __init__(
            self,
            remove_clients: bool,
            begin_removing_from_round: int,
            alpha: float,
            K: int,
            tau: float,
            edge_rule: str,
            neighbor_cap: int,
            graph_static: bool,
            convergence_eps: float,
            strategy_history: SimulationStrategyHistory,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.remove_clients = remove_clients
        self.begin_removing_from_round = begin_removing_from_round

        self.alpha = alpha
        self.K = K
        self.tau = tau
        self.edge_rule = edge_rule
        self.neighbor_cap = neighbor_cap
        self.graph_static = graph_static
        self.convergence_eps = convergence_eps

        self.strategy_history = strategy_history

        self.current_round = 0
        self.removed_client_ids = set()

        self.self_scores: Dict[str, float] = {}
        self.trust_scores: Dict[str, float] = {}

        self._adjacency_matrix: Optional[np.ndarray] = None
        self._last_client_order: List[str] = []

    def _flatten_params(self, fit_res_list: List[Tuple[ClientProxy, FitRes]]) -> Tuple[List[str], List[np.ndarray]]:
        client_ids = []
        tensors = []
        for client_proxy, fit_res in fit_res_list:
            client_ids.append(client_proxy.cid)
            client_params = fl.common.parameters_to_ndarrays(fit_res.parameters)
            flattened = [np.ravel(arr) for arr in client_params]
            param_tensor = np.concatenate(flattened)
            tensors.append(param_tensor)
        return client_ids, tensors

    def _compute_self_scores(self, tensors: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        X = np.array(tensors)
        try:
            from sklearn.cluster import KMeans
            from sklearn.preprocessing import MinMaxScaler
            kmeans = KMeans(n_clusters=1, init='k-means++').fit(X)
            distances = kmeans.transform(X)
            scaler = MinMaxScaler()
            scaler.fit(distances)
            normalized = scaler.transform(distances)
            self_scores = 1.0 - normalized[:, 0]
            return self_scores, distances[:, 0]
        except Exception:
            centroid = X.mean(axis=0)
            dists = np.linalg.norm(X - centroid, axis=1)
            d_min, d_max = dists.min(), dists.max()
            if d_max - d_min > 0:
                normalized = (dists - d_min) / (d_max - d_min)
            else:
                normalized = np.zeros_like(dists)
            self_scores = 1.0 - normalized
            return self_scores, dists

    def _build_similarity_adjacency(self, tensors: List[np.ndarray]) -> np.ndarray:
        n = len(tensors)
        D = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            for j in range(n):
                if i == j:
                    D[i, j] = 0.0
                else:
                    diff = tensors[i] - tensors[j]
                    D[i, j] = np.sqrt(np.dot(diff, diff))
        sigma = np.mean(D[D > 0]) if np.any(D > 0) else 1.0
        W = np.exp(-D / sigma)
        np.fill_diagonal(W, 0.0)
        if self.neighbor_cap and self.neighbor_cap > 0:
            for i in range(n):
                idx_sorted = np.argsort(W[i])[::-1]
                keep = idx_sorted[:self.neighbor_cap]
                mask = np.ones(n, dtype=bool)
                mask[keep] = False
                W[i][mask] = 0.0
        row_sums = W.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0.0] = 1.0
        W = W / row_sums
        return W

    def _ensure_adjacency(self, tensors: List[np.ndarray]) -> None:
        if self._adjacency_matrix is None or not self.graph_static:
            self._adjacency_matrix = self._build_similarity_adjacency(tensors)

    def _propagate_trust(self, self_vec: np.ndarray, init_trust: Optional[np.ndarray] = None) -> np.ndarray:
        t = init_trust.copy() if init_trust is not None else self_vec.copy()
        for _ in range(self.K):
            t_next = self.alpha * self_vec + (1.0 - self.alpha) * self._adjacency_matrix.T.dot(t)
            if np.linalg.norm(t_next - t) < self.convergence_eps:
                t = t_next
                break
            t = t_next
        t = np.clip(t, 0.0, 1.0)
        return t

    def aggregate_fit(
            self,
            server_round: int,
            results: List[Tuple[ClientProxy, FitRes]],
            failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:

        self.current_round += 1

        if not results:
            return super().aggregate_fit(server_round, results, failures)

        aggregate_clients = []
        for result in results:
            client_id = result[0].cid
            if client_id not in self.removed_client_ids:
                aggregate_clients.append(result)

        if not aggregate_clients:
            return super().aggregate_fit(server_round, results, failures)

        client_ids, tensors = self._flatten_params(aggregate_clients)
        self._last_client_order = client_ids

        self_vec, abs_dists = self._compute_self_scores(tensors)
        self._ensure_adjacency(tensors)

        init_trust_vec = None
        if all(cid in self.trust_scores for cid in client_ids):
            init_trust_vec = np.array([self.trust_scores[cid] for cid in client_ids])

        trust_vec = self._propagate_trust(self_vec, init_trust=init_trust_vec)

        for i, cid in enumerate(client_ids):
            self.self_scores[cid] = float(self_vec[i])
            self.trust_scores[cid] = float(trust_vec[i])

            self.strategy_history.insert_single_client_history_entry(
                current_round=self.current_round,
                client_id=int(cid),
                removal_criterion=float(trust_vec[i]),
                absolute_distance=float(abs_dists[i])
            )

        weights = trust_vec.copy()
        total = weights.sum()
        if total > 0:
            weights = weights / total
        else:
            weights = np.ones_like(weights) / len(weights)

        layer_arrays = [parameters_to_ndarrays(fit_res.parameters) for _, fit_res in aggregate_clients]
        aggregated_layers = []
        for layers in zip(*layer_arrays):
            stacked = np.stack(layers, axis=0)
            weighted = np.tensordot(weights, stacked, axes=(0, 0))
            aggregated_layers.append(weighted)

        aggregated_parameters = ndarrays_to_parameters(aggregated_layers)

        self.strategy_history.insert_round_history_entry(removal_threshold=self.tau)

        return aggregated_parameters, {}

    def configure_fit(self, server_round: int, parameters: Parameters, client_manager):
        available_clients = client_manager.all()

        if self.current_round <= self.begin_removing_from_round - 1:
            fit_ins = fl.common.FitIns(parameters, {})
            return [(client, fit_ins) for client in available_clients.values()]

        client_trusts = {client_id: self.trust_scores.get(client_id, 0.0) for client_id in available_clients.keys()}

        if self.remove_clients:
            for client_id, trust in client_trusts.items():
                if trust < self.tau and client_id not in self.removed_client_ids:
                    self.removed_client_ids.add(client_id)

        self.strategy_history.update_client_participation(
            current_round=self.current_round, removed_client_ids=self.removed_client_ids
        )

        sorted_client_ids = sorted(client_trusts, key=client_trusts.get, reverse=True)
        fit_ins = fl.common.FitIns(parameters, {})
        return [(available_clients[cid], fit_ins) for cid in sorted_client_ids if cid in available_clients]

    def aggregate_evaluate(
            self,
            server_round: int,
            results: List[Tuple[ClientProxy, EvaluateRes]],
            failures: List[Tuple[Union[ClientProxy, EvaluateRes], BaseException]]
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:

        logging.info('\n' + '-' * 50 + f'AGGREGATION ROUND {server_round}' + '-' * 50)
        for client_result in results:
            cid = client_result[0].cid
            accuracy_matrix = client_result[1].metrics
            accuracy_matrix['cid'] = cid

            self.strategy_history.insert_single_client_history_entry(
                client_id=int(cid),
                current_round=self.current_round,
                accuracy=accuracy_matrix['accuracy']
            )

        if not results:
            return None, {}

        aggregate_value = []
        number_of_clients_in_loss_calc = 0

        for client_metadata, evaluate_res in results:
            self.strategy_history.insert_single_client_history_entry(
                client_id=int(client_metadata.cid),
                current_round=self.current_round,
                loss=evaluate_res.loss
            )

            if client_metadata.cid not in self.removed_client_ids:
                aggregate_value.append((evaluate_res.num_examples, evaluate_res.loss))
                number_of_clients_in_loss_calc += 1

        loss_aggregated = weighted_loss_avg(aggregate_value)
        self.strategy_history.insert_round_history_entry(loss_aggregated=loss_aggregated)

        metrics_aggregated = {}
        logging.info(
            f'Round: {server_round} '
            f'Number of aggregated clients: {number_of_clients_in_loss_calc} '
            f'Aggregated loss: {loss_aggregated} '
        )

        return loss_aggregated, metrics_aggregated
