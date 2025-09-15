from typing import Dict, Any, List, Optional
from loguru import logger
from datetime import datetime
import json
from pathlib import Path


class CheckpointManager:
    """Manages model checkpoints and training history."""

    def __init__(self, checkpoint_dir: str, algorithm_name: str):
        self.algorithm_name = algorithm_name
        self.checkpoint_dir = Path(checkpoint_dir) / algorithm_name
        
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Training history
        self.training_history = []
        self.best_reward = float('-inf')
        self.best_episode = 0

        logger.info(
            f"CheckpointManager initialized for {algorithm_name} at {checkpoint_dir}")

    def save_checkpoint(self, algorithm, episode: int, metrics: Dict[str, Any]):
        """Save algorithm checkpoint with metadata."""
        try:
            # Save latest checkpoint
            latest_path = self.checkpoint_dir / "latest_checkpoint"
            if self.algorithm_name == 'q_learning':
                latest_path = latest_path.with_suffix('.pkl')
            else:
                latest_path = latest_path.with_suffix('.pth')

            algorithm.save(str(latest_path))

            # Save episode-specific checkpoint
            episode_path = self.checkpoint_dir / f"episode_{episode:04d}"
            if self.algorithm_name == 'q_learning':
                episode_path = episode_path.with_suffix('.pkl')
            else:
                episode_path = episode_path.with_suffix('.pth')

            algorithm.save(str(episode_path))

            # Check if this is the best model
            episode_reward = metrics.get('total_reward', float('-inf'))
            if episode_reward > self.best_reward:
                self.best_reward = episode_reward
                self.best_episode = episode

                # Save best checkpoint
                best_path = self.checkpoint_dir / "best_checkpoint"
                if self.algorithm_name == 'q_learning':
                    best_path = best_path.with_suffix('.pkl')
                else:
                    best_path = best_path.with_suffix('.pth')

                algorithm.save(str(best_path))
                logger.info(
                    f"New best model saved! Episode {episode}, Reward: {episode_reward:.2f}")

            # Save metadata
            self._save_metadata(episode, metrics)

            logger.info(f"Checkpoint saved for episode {episode}")

        except Exception as e:
            logger.error(f"Error saving checkpoint: {e}")

    def load_checkpoint(self, algorithm, checkpoint_type: str = "latest") -> Optional[Dict[str, Any]]:
        """Load algorithm checkpoint.

        Args:
            algorithm: Algorithm instance to load into
            checkpoint_type: "latest", "best", or specific episode number

        Returns:
            Metadata dictionary if successful, None otherwise
        """
        try:
            if checkpoint_type == "latest":
                checkpoint_path = self.checkpoint_dir / "latest_checkpoint"
            elif checkpoint_type == "best":
                checkpoint_path = self.checkpoint_dir / "best_checkpoint"
            elif checkpoint_type.startswith("episode_"):
                checkpoint_path = self.checkpoint_dir / checkpoint_type
            else:
                # Assume it's an episode number
                try:
                    episode_num = int(checkpoint_type)
                    checkpoint_path = self.checkpoint_dir / \
                        f"episode_{episode_num:04d}"
                except ValueError:
                    logger.error(f"Invalid checkpoint type: {checkpoint_type}")
                    return None

            # Add appropriate extension
            if self.algorithm_name == 'q_learning':
                checkpoint_path = checkpoint_path.with_suffix('.pkl')
            else:
                checkpoint_path = checkpoint_path.with_suffix('.pth')

            if not checkpoint_path.exists():
                logger.warning(f"Checkpoint not found: {checkpoint_path}")
                return None

            algorithm.load(str(checkpoint_path))
            logger.info(f"Loaded checkpoint: {checkpoint_path}")

            # Load metadata
            return self._load_metadata()

        except Exception as e:
            logger.error(f"Error loading checkpoint: {e}")
            return None

    def _save_metadata(self, episode: int, metrics: Dict[str, Any]):
        """Save training metadata."""
        metadata = {
            'episode': episode,
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics,
            'best_reward': self.best_reward,
            'best_episode': self.best_episode
        }

        self.training_history.append(metadata)

        # Save to file
        metadata_path = self.checkpoint_dir / "training_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump({
                'algorithm': self.algorithm_name,
                'best_reward': self.best_reward,
                'best_episode': self.best_episode,
                'history': self.training_history
            }, f, indent=2)

    def _load_metadata(self) -> Optional[Dict[str, Any]]:
        """Load training metadata."""
        metadata_path = self.checkpoint_dir / "training_metadata.json"
        if not metadata_path.exists():
            return None

        try:
            with open(metadata_path, 'r') as f:
                data = json.load(f)
                self.training_history = data.get('history', [])
                self.best_reward = data.get('best_reward', float('-inf'))
                self.best_episode = data.get('best_episode', 0)
                return data
        except Exception as e:
            logger.error(f"Error loading metadata: {e}")
            return None

    def get_training_history(self) -> List[Dict[str, Any]]:
        """Get training history."""
        return self.training_history.copy()

    def cleanup_old_checkpoints(self, keep_last_n: int = 5):
        """Remove old episode checkpoints, keeping only the last N."""
        try:
            pattern = f"episode_*.{'pkl' if self.algorithm_name == 'q_learning' else 'pth'}"
            episode_files = sorted(self.checkpoint_dir.glob(pattern))

            if len(episode_files) > keep_last_n:
                files_to_remove = episode_files[:-keep_last_n]
                for file_path in files_to_remove:
                    file_path.unlink()
                    logger.debug(f"Removed old checkpoint: {file_path}")

                logger.info(
                    f"Cleaned up {len(files_to_remove)} old checkpoints")

        except Exception as e:
            logger.error(f"Error cleaning up checkpoints: {e}")
