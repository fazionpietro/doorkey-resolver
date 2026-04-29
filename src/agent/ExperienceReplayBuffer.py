import collections
import typing
import numpy as np

_field_names = ["state", "action", "reward", "next_state", "done"]
Experience = collections.namedtuple("Experience", field_names=_field_names)


class ExperienceReplayBuffer:

    def __init__(
        self,
        batch_size: int,
        buffer_size: int,
        alpha: float,
        random_state: np.random.RandomState,
    ) -> None:
        self._batch_size = batch_size
        self._buffer_size = buffer_size
        self._alpha = alpha
        self._random_state = (
            np.random.RandomState() if random_state is None else random_state
        )
        self._buffer = np.empty(
            self.buffer_size,
            dtype=[("priority", np.float32), ("experience", Experience)],
        )
        self._buffer_length = 0
        self._ptr = 0

    def __len__(self) -> int:
        """Current number of prioritized experience tuple stored in buffer."""
        return self._buffer_length

    @property
    def alpha(self):
        """Strength of prioritized sampling."""
        return self._alpha

    @property
    def batch_size(self) -> int:
        """Number of experience samples per training batch."""
        return self._batch_size

    @property
    def buffer_size(self) -> int:
        """Maximum number of prioritized experience tuples stored in buffer."""
        return self._buffer_size

    def is_empty(self) -> bool:
        """True if the buffer is empty; False otherwise."""
        return self._buffer_length == 0

    def is_full(self) -> bool:
        """True if the buffer is full; False otherwise."""
        return self._buffer_length == self._buffer_size

    def add(self, experience: Experience) -> None:
        """Add a new experience to memory."""
        priority = (
            1.0
            if self.is_empty()
            else self._buffer[: self._buffer_length]["priority"].max()
        )
        self._buffer[self._ptr] = (priority, experience)

        self._ptr = (self._ptr + 1) % self._buffer_size
        self._buffer_length = min(self._buffer_length + 1, self._buffer_size)

    def sample(self, beta: float) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sample a batch of experiences from memory."""
        # use sampling scheme to determine which experiences to use for learning
        ps = self._buffer[: self._buffer_length]["priority"]
        sampling_probs = ps**self._alpha / np.sum(ps**self._alpha)
        idxs = self._random_state.choice(
            np.arange(ps.size), size=self._batch_size, replace=True, p=sampling_probs
        )

        # select the experiences and compute sampling weights
        experiences = self._buffer["experience"][idxs]
        weights = (self._buffer_length * sampling_probs[idxs]) ** -beta
        normalized_weights = weights / weights.max()

        return idxs, experiences, normalized_weights

    def update_priorities(self, idxs: np.ndarray, priorities: np.ndarray) -> None:
        """Update the priorities associated with particular experiences."""
        self._buffer["priority"][idxs] = priorities
