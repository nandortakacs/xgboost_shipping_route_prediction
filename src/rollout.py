
import numpy as np
from shipping_route_predictor.env import EnvGraph, Action
from shipping_route_predictor.utils import discrete_frechet_distance
from shipping_route_predictor.config import RolloutConfig, TrajectorySpec, RolloutType
import xgboost as xgb

class Rollout:
	"""Handles greedy and CISC rollouts for XGBoost models on EnvGraph."""
	def __init__(self, rollout_cfg: RolloutConfig, model: xgb.XGBClassifier, env: EnvGraph):
		self.cfg = rollout_cfg
		self.model = model
		self.env = env
		self.H = env.H
		self.W = env.W

	def run(self, traj: TrajectorySpec, rollout_type, *, max_steps: int | None = None):
		"""Run rollout for a single trajectory with specified rollout type (greedy or CISC).

		Parameters
		----------
		max_steps : int, optional
			Override for ``cfg.max_rollout_steps``.  When *None* the
			config default is used.
		"""
		if max_steps is None:
			max_steps = self.cfg.max_rollout_steps
		cisc = self.cfg.model.cisc
		temperature = cisc.temperature if cisc else 1.0
		if rollout_type == RolloutType.GREEDY:
			return self._greedy_rollout(traj, temperature=temperature, max_steps=max_steps)
		elif rollout_type == RolloutType.CISC:
			return self.cisc_rollout(traj, max_steps=max_steps)
		else:
			raise ValueError(f"Unknown rollout type: {rollout_type}")

	def run_route(self, traj: TrajectorySpec, rollout_type, *, max_steps: int | None = None):
		"""Run rollout and return standardized ``(result, latlon_path)``.

		Wraps :meth:`run` and :meth:`grid_to_latlon_path` into a single call
		that both inference and eval can use directly.

		Returns ``(None, None)`` on failure.
		"""
		try:
			result = self.run(traj, rollout_type, max_steps=max_steps)
		except Exception:
			return None, None
		latlon = self.grid_to_latlon_path(result["grid_path"]) if result and result.get("grid_path") else None
		return result, latlon

	def _greedy_rollout(self, traj: TrajectorySpec, temperature: float = 1.0, max_steps: int | None = None):
		"""Run a greedy rollout (argmax at each step, probs scaled by temperature)."""
		if max_steps is None:
			max_steps = self.cfg.max_rollout_steps
		self.env.reset(traj)
		grid_path = [self.env.position]
		step_probs = []
		for _ in range(max_steps):
			feats = self.env.build_input_features().reshape(1, -1)
			base_probs = self.model.predict_proba(feats)[0]  # softmax at T=1
			probs = np.exp(np.log(base_probs + 1e-12) / temperature)
			valid = self.env.valid_actions(*self.env.position)
			probs[~valid] = 0.0
			if probs.sum() == 0:
				break  # stuck — no valid moves
			probs /= probs.sum()
			action_idx = int(np.argmax(probs))
			step_probs.append(float(probs[action_idx]))
			action = Action.from_index(action_idx)
			self.env.step(action)
			grid_path.append(self.env.position)
			if self.env.reached_goal():
				break
		return {
			"grid_path": grid_path,
			"reached_goal": self.env.reached_goal(),
			"is_greedy": True,
			"step_probs": step_probs,
			"mean_prob": float(np.mean(step_probs)) if step_probs else 0.0,
		}

	def cisc_rollout(self, traj: TrajectorySpec, *, max_steps: int | None = None):
		"""Generate N rollouts and select the best by CISC scoring."""
		cisc_config = self.cfg.model.cisc
		rollouts = self._generate_rollouts(traj, cisc_config.n_rollouts, cisc_config.temperature, max_steps=max_steps)
		selected, _score = self.select_cisc(rollouts)
		return selected

	def select_cisc(self, rollouts, alpha=None, beta=None):
		"""Select the best rollout from a pre-generated set using CISC scoring.

		Parameters
		----------
		rollouts : list[dict]
			Pre-generated rollouts (greedy + stochastic).
		alpha : float, optional
			Weight for self-consistency term. Defaults to config value.
		beta : float, optional
			Weight for confidence term. Defaults to config value.

		Returns
		-------
		selected : dict
			The best rollout.
		score : float
			CISC score of the selected rollout (nan if no successful rollouts).
		"""
		cisc = self.cfg.model.cisc
		if alpha is None:
			alpha = cisc.alpha
		if beta is None:
			beta = cisc.beta

		successful = [r for r in rollouts if r["reached_goal"]]
		if not successful:
			return rollouts[0], float("nan")

		dist_matrix = self._calculate_distance_matrix(successful)
		mean_dist = self._calculate_mean_distance(dist_matrix)
		confidence = self._calculate_confidence_scores(successful)
		scores = -alpha * self._z_score(mean_dist) + beta * self._z_score(confidence)
		best_idx = int(np.argmax(scores))
		return successful[best_idx], float(scores[best_idx])

	def _generate_rollouts(self, traj, n, temperature, *, max_steps: int | None = None):
		"""Generate one greedy and (n-1) stochastic rollouts."""
		rollouts = [self._greedy_rollout(traj, temperature=temperature, max_steps=max_steps)]
		for _ in range(n-1):
			rollouts.append(self._stochastic_rollout(traj, temperature=temperature, max_steps=max_steps))
		return rollouts

	def _stochastic_rollout(self, traj: TrajectorySpec, temperature: float = 1.0, max_steps: int | None = None):
		"""Run a stochastic rollout (sample from temperature-scaled probabilities)."""
		if max_steps is None:
			max_steps = self.cfg.max_rollout_steps
		self.env.reset(traj)
		grid_path = [self.env.position]
		step_probs = []
		for _ in range(max_steps):
			feats = self.env.build_input_features().reshape(1, -1)
			base_probs = self.model.predict_proba(feats)[0]  # softmax at T=1
			probs = np.exp(np.log(base_probs + 1e-12) / temperature)  # re-scale to desired T
			valid = self.env.valid_actions(*self.env.position)
			probs[~valid] = 0.0
			if probs.sum() == 0:
				break  # stuck — no valid moves
			probs /= probs.sum()
			action_idx = int(np.random.choice(len(Action), p=probs))
			step_probs.append(float(probs[action_idx]))
			action = Action.from_index(action_idx)
			self.env.step(action)
			grid_path.append(self.env.position)
			if self.env.reached_goal():
				break
		return {
			"grid_path": grid_path,
			"reached_goal": self.env.reached_goal(),
			"is_greedy": False,
			"step_probs": step_probs,
			"mean_prob": float(np.mean(step_probs)) if step_probs else 0.0}
	
	
	def _calculate_distance_matrix(self, rollouts):
		"""Compute pairwise Fréchet distance matrix for rollouts."""
		paths = [self.grid_to_latlon_path(r["grid_path"]) for r in rollouts]
		n = len(paths)
		matrix = np.zeros((n, n))
		for i in range(n):
			for j in range(n):
				if i == j:
					matrix[i, j] = 0.0
				else:
					matrix[i, j] = discrete_frechet_distance(paths[i], paths[j])
		return matrix

	def _calculate_mean_distance(self, similarity_matrix):
		"""Mean Fréchet distance to other rollouts for each rollout."""
		n = similarity_matrix.shape[0]
		return np.array([
			np.mean([similarity_matrix[i, j] for j in range(n) if i != j])
			for i in range(n)
		])

	def _calculate_confidence_scores(self, rollouts):
		"""Return mean probability for each rollout as confidence score."""
		return np.array([r["mean_prob"] for r in rollouts])

	def _z_score(self, arr):
		arr = np.asarray(arr)
		return (arr - arr.mean()) / (arr.std() + 1e-8)

	def grid_to_latlon_path(self, grid_path):
		"""Convert a path of grid coordinates (row, col) to (lat, lon)."""
		return [self.env.grid_indices_to_latlon(rc) for rc in grid_path]
