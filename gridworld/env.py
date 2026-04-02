import random
from typing import Optional, Tuple
import numpy as np


class GridWorld:
    """Grid world where task = hidden goal position."""
    def __init__(
        self,
        size: int = 5,
        max_steps: int = 15,
        step_penalty: float = -0.1,
        goal_reward: float = 1.0,
    ) -> None:
        self.size = int(size)
        self.max_steps = int(max_steps)
        self.step_penalty = float(step_penalty)
        self.goal_reward = float(goal_reward)

        self.start = (0, 0)
        self.action_dim = 4  # up, right, down, left
        self.observation_dim = 2

        self._rng = random.Random()
        self.goal = self.sample_task()
        self.pos = self.start
        self.steps = 0

    def seed(self, seed: int) -> None:
        self._rng.seed(seed)
        np.random.seed(seed)

    def sample_task(self) -> Tuple[int, int]:
        choices = [
            (x, y)
            for x in range(self.size)
            for y in range(self.size)
            if (x, y) != self.start and (x, y) not in [
                (self.start[0] + dx, self.start[1] + dy)
                for dx in [-1, 0, 1]
                for dy in [-1, 0, 1]
                if (dx, dy) != (0, 0)
            ]
        ]
        return self._rng.choice(choices)

#     def reset_task(self, task: Optional[Tuple[int, int]] = None) -> Tuple[int, int]:
#         if task is None:
#             self.goal = self.sample_task()
#         else:
#             self.goal = (int(task[0]), int(task[1]))
#         return self.goal

#     def reset(self) -> np.ndarray:
#         self.pos = self.start
#         self.steps = 0
#         return self._obs()

#     def _obs(self) -> np.ndarray:
#         if self.size <= 1:
#             return np.zeros(2, dtype=np.float32)
#         return np.array(
#             [self.pos[0] / (self.size - 1), self.pos[1] / (self.size - 1)],
#             dtype=np.float32,
#         )

#     def step(self, action: int):
#         x, y = self.pos
#         if action == 0:  # up
#             y = min(y + 1, self.size - 1)
#         elif action == 1:  # right
#             x = min(x + 1, self.size - 1)
#         elif action == 2:  # down
#             y = max(y - 1, 0)
#         elif action == 3:  # left
#             x = max(x - 1, 0)
#         else:
#             raise ValueError(f"Invalid action: {action}")

#         self.pos = (x, y)
#         self.steps += 1

#         reached_goal = self.pos == self.goal
#         done = reached_goal or self.steps >= self.max_steps
#         reward = self.goal_reward if reached_goal else self.step_penalty

#         info = {
#             "goal": self.goal,
#             "reached_goal": reached_goal,
#         }
#         return self._obs(), float(reward), bool(done), info
