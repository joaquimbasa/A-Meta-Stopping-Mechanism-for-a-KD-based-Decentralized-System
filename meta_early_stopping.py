"""
Meta Early Stopping Module for Decentralized Knowledge Distillation

This module implements the meta early stopping criteria formalized in final_formalization.tex
It provides a clean interface for monitoring local and meta-validation losses and determining
when each client should stop its iterative training process.

Author: Based on the formalization in Section 3.2
"""

import torch
import numpy as np
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
import json


@dataclass
class MetaStoppingConfig:
    """
    Configuration for meta early stopping hyperparameters

    Attributes:
        P_patience (int): Number of iterations to wait for improvement before stopping (default: 5)
        epsilon_tol (float): Absolute tolerance threshold for patience criterion (default: 0.001)
        n_div (int): Number of consecutive iterations for sustained divergence detection (default: 3)
        epsilon_conv (float): Convergence threshold for normalized improvement rate (default: 0.01)
        n_conv (int): Number of consecutive iterations for convergence criterion (default: 3)
        t_max (int): Maximum number of iterations (default: 10)
        use_convergence (bool): Whether to enable convergence criterion C3 (default: True)
        save_history (bool): Whether to save loss history for analysis (default: True)
    """
    P_patience: int = 5
    epsilon_tol: float = 0.01
    n_div: int = 3
    epsilon_conv: float = 0.01  # 1% of best loss
    n_conv: int = 3
    t_max: int = 10
    use_convergence: bool = True
    save_history: bool = True


class MetaEarlyStopping:
    """
    Meta Early Stopping mechanism for a single client in decentralized KD

    This class implements the stopping criteria defined in Equations 2-8 of final_formalization.tex:
    - C2: Sustained Local-Meta Divergence (Eq. 3-4) - KEY CRITERION
    - C3: Convergence Rate Monitoring (Eq. 5-8)
    - C5: Maximum Iterations

    The mechanism monitors both local validation loss (client's own data) and
    meta-validation loss (average over neighbors' data) to determine optimal stopping point.
    """

    def __init__(self, client_id: int, config: MetaStoppingConfig):
        """
        Initialize meta early stopping for a client

        Args:
            client_id: Identifier for the client
            config: Configuration object with hyperparameters
        """
        self.client_id = client_id
        self.config = config

        # Iteration counter
        self.t = 0

        # Best performance tracking (for Eq. 2)
        self.L_val_best = float('inf')
        self.L_meta_best = float('inf')
        self.t_best = 0

        # Patience counter for C1
        self.patience_counter = 0

        # Divergence counter for C2 (Eq. 4)
        self.div_counter = 0

        # Convergence counter for C3 (Eq. 8)
        self.conv_counter = 0

        # Previous iteration values (for rate computation in Eq. 5-6)
        self.L_val_prev = None
        self.L_meta_prev = None

        # Stopping decision
        self.should_stop = False
        self.stop_reason = None
        self.stop_iteration = None

        # History for analysis
        if self.config.save_history:
            self.history = {
                'iteration': [],
                'L_val': [],
                'L_meta': [],
                'L_val_best': [],
                'L_meta_best': [],
                'divergence_indicator': [],
                'patience_counter': [],
                'div_counter': [],
                'conv_counter': []
            }

    def update(self, L_val: float, L_meta: float) -> Tuple[bool, Optional[str]]:
        """
        Update stopping criteria with current iteration's losses

        This is the main function to call after each training iteration.

        Args:
            L_val: Local validation loss on client's own validation dataset
            L_meta: Meta-validation loss (average over neighbors' datasets)

        Returns:
            Tuple of (should_stop, reason):
                - should_stop (bool): True if training should stop
                - reason (str or None): Explanation of stopping criterion that triggered

        Example:
            >>> stopper = MetaEarlyStopping(client_id=0, config=MetaStoppingConfig())
            >>> should_stop, reason = stopper.update(L_val=0.45, L_meta=0.52)
            >>> if should_stop:
            >>>     print(f"Stopping at iteration {stopper.t}: {reason}")
        """
        self.t += 1

        # Save history
        if self.config.save_history:
            self.history['iteration'].append(self.t)
            self.history['L_val'].append(L_val)
            self.history['L_meta'].append(L_meta)

        # Track best performance independently for each metric
        # This follows the formalization: both are independent minimums
        improved = False

        if L_val < self.L_val_best:
            self.L_val_best = L_val
            self.t_best = self.t  # Record iteration with best LOCAL validation
            improved = True

        if L_meta < self.L_meta_best:
            self.L_meta_best = L_meta
            # Note: We use local validation for t_best, but track meta-best independently


        # Best Combined Performance (Can be the most theoretically sound)

        # # Update both only when BOTH improve, OR use a combined metric
        # if L_val < self.L_val_best AND L_meta < self.L_meta_best:
        #     self.L_val_best = L_val
        #     self.L_meta_best = L_meta
        #     self.t_best = self.t
        # elif L_val < self.L_val_best:
        #     self.L_val_best = L_val
        # elif L_meta < self.L_meta_best:
        #     self.L_meta_best = L_meta

        # Update patience counter based on LOCAL validation improvement
        # (since we're primarily optimizing for local performance)
        """if improved:
            self.patience_counter = 0
        else:
            self.patience_counter += 1"""
        if improved:
            # Loss improved - reset patience counter
            self.patience_counter = 0
        elif (L_val - self.L_val_best)/self.L_val_best > self.config.epsilon_tol:
            # Loss degraded beyond tolerance - increment patience
            self.patience_counter += 1
        # else: loss within tolerance of best - don't increment patience

        # Save best values to history
        if self.config.save_history:
            self.history['L_val_best'].append(self.L_val_best)
            self.history['L_meta_best'].append(self.L_meta_best)
            self.history['patience_counter'].append(self.patience_counter)

        # Check all stopping criteria
        stop, reason = self._check_criteria(L_val, L_meta)

        # Update previous values for next iteration (needed for Eq. 5-6)
        self.L_val_prev = L_val
        self.L_meta_prev = L_meta

        if stop:
            self.should_stop = True
            self.stop_reason = reason
            self.stop_iteration = self.t

        return stop, reason

    def _check_criteria(self, L_val: float, L_meta: float) -> Tuple[bool, Optional[str]]:
        """
        Check all stopping criteria and return decision

        Implements the composite stopping function from Eq. 9 in final_formalization.tex

        Args:
            L_val: Current local validation loss
            L_meta: Current meta-validation loss

        Returns:
            Tuple of (should_stop, reason)
        """
        # C1: Patience Criterion (Eq. 2 in formalization)
        # Stop if loss has degraded beyond tolerance for P_patience consecutive iterations
        # (The tolerance check is done when incrementing patience_counter)
        if self.patience_counter >= self.config.P_patience:
            return True, f"C1: Patience exceeded - loss degraded beyond tolerance for {self.config.P_patience} consecutive iterations"

        # C2: Sustained Local-Meta Divergence (Eq. 3-4) - THE KEY EGOCENTRIC CRITERION
        if self._check_divergence(L_val, L_meta):
            return True, "C2: Sustained divergence - local loss increases while meta-loss stable/decreases"

        # C3: Convergence Rate Monitoring (Eq. 5-8)
        if self.config.use_convergence and self._check_convergence(L_val, L_meta):
            return True, "C3: Convergence achieved - improvement rate below threshold for both losses"

        # C5: Maximum Iterations
        if self.t >= self.config.t_max:
            return True, "C5: Maximum iterations reached"

        return False, None

    def _check_divergence(self, L_val: float, L_meta: float) -> bool:
        """
        C2: Check for sustained local-meta divergence (Eq. 3-4)

        This is the KEY innovation of the meta early stopping approach.

        Divergence indicator (Eq. 3):
            Div^{k,t} = 1 if L_val increases AND L_meta stable/decreases
            Div^{k,t} = 0 otherwise

        Sustained divergence (Eq. 4):
            Stop if sum of last n_div indicators equals n_div
            (i.e., divergence detected for n_div consecutive iterations)

        Args:
            L_val: Current local validation loss
            L_meta: Current meta-validation loss

        Returns:
            True if sustained divergence detected, False otherwise
        """
        if self.L_val_prev is None or self.L_meta_prev is None:
            # First iteration - no previous values to compare
            if self.config.save_history:
                self.history['divergence_indicator'].append(0)
                self.history['div_counter'].append(0)
            return False

        # Compute divergence indicator (Eq. 3)
        # Div = 1 if local worsens AND meta improves/stable
        local_increases = L_val > self.L_val_prev
        meta_stable_or_decreases = L_meta <= self.L_meta_prev

        divergence_indicator = 1 if (local_increases and meta_stable_or_decreases) else 0

        # Update divergence counter
        if divergence_indicator == 1:
            self.div_counter += 1
        else:
            self.div_counter = 0  # Reset if divergence not detected

        # Save to history
        if self.config.save_history:
            self.history['divergence_indicator'].append(divergence_indicator)
            self.history['div_counter'].append(self.div_counter)

        # Check sustained divergence (Eq. 4)
        # Stop if divergence detected for n_div consecutive iterations
        return self.div_counter >= self.config.n_div

    def _check_convergence(self, L_val: float, L_meta: float) -> bool:
        """
        C3: Check convergence rate monitoring (Eq. 5-8)

        Improvement rates (Eq. 5-6):
            r_local^{k,t} = L_val^{k,t-1} - L_val^{k,t}
            r_meta^{k,t} = L_meta^{k,t-1} - L_meta^{k,t}

        Normalized convergence criterion (Eq. 8 - Option C):
            |r_local| / L_val_best < epsilon_conv
            AND
            |r_meta| / L_meta_best < epsilon_conv

        Must hold for n_conv consecutive iterations.

        Args:
            L_val: Current local validation loss
            L_meta: Current meta-validation loss

        Returns:
            True if converged (both losses changing slowly), False otherwise
        """
        if self.L_val_prev is None or self.L_meta_prev is None:
            # First iteration - cannot compute rates
            if self.config.save_history:
                self.history['conv_counter'].append(0)
            return False

        # Compute improvement rates (Eq. 5-6)
        r_local = self.L_val_prev - L_val
        r_meta = self.L_meta_prev - L_meta

        # Normalize by best losses (Eq. 8 - Option C from formalization)
        # This makes the threshold scale-invariant
        if self.L_val_best > 0:
            r_local_normalized = abs(r_local) / self.L_val_best
        else:
            r_local_normalized = abs(r_local)

        if self.L_meta_best > 0:
            r_meta_normalized = abs(r_meta) / self.L_meta_best
        else:
            r_meta_normalized = abs(r_meta)

        # Check convergence criterion (both rates below threshold)
        converged_this_iter = (r_local_normalized < self.config.epsilon_conv and
                               r_meta_normalized < self.config.epsilon_conv)

        # Update convergence counter
        if converged_this_iter:
            self.conv_counter += 1
        else:
            self.conv_counter = 0  # Reset if not converged

        # Save to history
        if self.config.save_history:
            self.history['conv_counter'].append(self.conv_counter)

        # Check sustained convergence (must hold for n_conv consecutive iterations)
        return self.conv_counter >= self.config.n_conv

    def get_best_iteration(self) -> int:
        """
        Get the iteration at which best validation loss was achieved

        Returns:
            Iteration number of best model (t_best^k from formalization)
        """
        return self.t_best

    def get_state(self) -> Dict:
        """
        Get current state of the stopping mechanism

        Returns:
            Dictionary with all state variables for debugging/logging
        """
        return {
            'client_id': self.client_id,
            'iteration': self.t,
            't_best': self.t_best,
            'L_val_best': self.L_val_best,
            'L_meta_best': self.L_meta_best,
            'patience_counter': self.patience_counter,
            'div_counter': self.div_counter,
            'conv_counter': self.conv_counter,
            'should_stop': self.should_stop,
            'stop_reason': self.stop_reason,
            'stop_iteration': self.stop_iteration
        }

    def save_history(self, filepath: str):
        """
        Save loss history to JSON file for analysis

        Args:
            filepath: Path to save JSON file
        """
        if not self.config.save_history:
            raise ValueError("History tracking is disabled. Set save_history=True in config.")

        # Convert numpy types to native Python types for JSON serialization
        history_serializable = {
            key: [float(v) if isinstance(v, (np.floating, np.integer)) else v
                  for v in values]
            for key, values in self.history.items()
        }

        # Add metadata
        output = {
            'client_id': self.client_id,
            'config': {
                'P_patience': self.config.P_patience,
                'epsilon_tol': self.config.epsilon_tol,
                'n_div': self.config.n_div,
                'epsilon_conv': self.config.epsilon_conv,
                'n_conv': self.config.n_conv,
                't_max': self.config.t_max
            },
            'final_state': self.get_state(),
            'history': history_serializable
        }

        with open(filepath, 'w') as f:
            json.dump(output, f, indent=2)

    def reset(self):
        """
        Reset all state variables (useful for multiple experiments)
        """
        self.t = 0
        self.L_val_best = float('inf')
        self.L_meta_best = float('inf')
        self.t_best = 0
        self.patience_counter = 0
        self.div_counter = 0
        self.conv_counter = 0
        self.L_val_prev = None
        self.L_meta_prev = None
        self.should_stop = False
        self.stop_reason = None
        self.stop_iteration = None

        if self.config.save_history:
            self.history = {
                'iteration': [],
                'L_val': [],
                'L_meta': [],
                'L_val_best': [],
                'L_meta_best': [],
                'divergence_indicator': [],
                'patience_counter': [],
                'div_counter': [],
                'conv_counter': []
            }


def compute_meta_validation_loss(
    model: torch.nn.Module,
    neighbor_val_loaders: list,
    criterion: torch.nn.Module,
    device: torch.device
) -> float:
    """
    Compute meta-validation loss (Eq. 1 from final_formalization.tex)

    L_meta^{k,t} = (1/|Phi_k|) * sum_{phi in Phi_k} L_val^{k,phi}

    This measures how well the client's model generalizes to neighbors' data distributions.

    Args:
        model: The client's model to evaluate
        neighbor_val_loaders: List of validation DataLoaders from neighboring clients
        criterion: Loss function (e.g., nn.CrossEntropyLoss())
        device: Device to run computations on

    Returns:
        Meta-validation loss (average validation loss over all neighbors)

    Example:
        >>> meta_loss = compute_meta_validation_loss(
        ...     model=student_model,
        ...     neighbor_val_loaders=[valloader_client0, valloader_client1],
        ...     criterion=nn.CrossEntropyLoss(),
        ...     device=device
        ... )
    """
    model.eval()
    total_loss = 0.0
    num_neighbors = len(neighbor_val_loaders)

    if num_neighbors == 0:
        raise ValueError("No neighbor validation loaders provided")

    with torch.no_grad():
        for neighbor_loader in neighbor_val_loaders:
            neighbor_loss = 0.0
            num_batches = 0

            for inputs, labels in neighbor_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                # Forward pass through second head
                _, outputs = model(inputs)

                # Compute loss
                loss = criterion(outputs, labels)
                neighbor_loss += loss.item()
                num_batches += 1

            # Average loss for this neighbor
            if num_batches > 0:
                neighbor_loss /= num_batches
                total_loss += neighbor_loss

    # Compute meta-validation loss (Eq. 1)
    meta_val_loss = total_loss / num_neighbors

    return meta_val_loss


# Example usage and testing
if __name__ == "__main__":
    """
    Example demonstrating how to use the MetaEarlyStopping class
    This recreates the example from the formalization document
    """

    print("=" * 70)
    print("Meta Early Stopping - Example Usage")
    print("=" * 70)

    # Create configuration
    config = MetaStoppingConfig(
        P_patience=5,
        epsilon_tol=0.01,
        n_div=3,
        epsilon_conv=0.01,
        n_conv=3,
        t_max=15
    )

    # Create stopper for client 0
    stopper = MetaEarlyStopping(client_id=0, config=config)

    # Simulated training trajectory (matching example from formalization)
    # (L_val, L_meta) at each iteration
    training_data = [
        (0.450, 0.520),  # iter 1 - both decreasing
        (0.420, 0.485),  # iter 2
        (0.395, 0.460),  # iter 3
        (0.385, 0.445),  # iter 4
        (0.382, 0.438),  # iter 5
        (0.381, 0.436),  # iter 6 - best local
        (0.383, 0.435),  # iter 7 - divergence starts (local up, meta down)
        (0.386, 0.434),  # iter 8 - divergence continues
        (0.390, 0.433),  # iter 9 - sustained divergence (3 iters) -> STOP
    ]

    print(f"\n{'Iter':<6} {'L_val':<8} {'L_meta':<8} {'t_best':<8} {'Div':<5} {'Stop':<6} {'Reason':<50}")
    print("-" * 100)

    for L_val, L_meta in training_data:
        should_stop, reason = stopper.update(L_val, L_meta)

        state = stopper.get_state()

        print(f"{state['iteration']:<6} {L_val:<8.3f} {L_meta:<8.3f} {state['t_best']:<8} "
              f"{state['div_counter']:<5} {should_stop!s:<6} {reason if reason else '':<50}")

        if should_stop:
            print("\n" + "=" * 100)
            print(f"STOPPING at iteration {state['iteration']}")
            print(f"Reason: {reason}")
            print(f"Best model from iteration {state['t_best']} with L_val = {state['L_val_best']:.3f}")
            print("=" * 100)
            break

    # Save history
    stopper.save_history('meta_stopping_history_example.json')
    print(f"\nHistory saved to meta_stopping_history_example.json")

    print("\n" + "=" * 70)
    print("Final State:")
    print("=" * 70)
    import json
    print(json.dumps(stopper.get_state(), indent=2))
