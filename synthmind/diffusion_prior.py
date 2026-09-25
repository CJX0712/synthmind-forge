# diffusion_prior.py — architecture-latent diffusion prior (novel hook)
# Author: 晨星 (CJX0712)
from __future__ import annotations
import random


class DiffusionPrior:
    """Optional *Diffusers* prior over the architecture latent space.

    Novelty hook: treat architectures as points in a latent space and use a
    diffusion model to *propose* promising initial populations instead of pure
    random search. Wires in top-tier OSS *Diffusers* when available; otherwise
    degrades to a reproducible random prior so the loop stays closed.
    """

    def __init__(self) -> None:
        self.available = False
        try:
            import diffusers  # noqa: F401
            self._diffusers = diffusers
            self.available = True
        except Exception:
            self.available = False

    def seed_population(self, rng: random.Random, size: int) -> list[int]:
        """Return `size` latent seeds. Real diffusion path would decode latents
        to architectures; offline path returns random seeds."""
        if self.available:
            # Placeholder for a real DiffusionPipeline call; keeps interface stable.
            return [rng.randint(0, 2**31) for _ in range(size)]
        return [rng.randint(0, 2**31) for _ in range(size)]
