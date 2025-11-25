"""Model-backed PokerAgent that uses a PyTorch policy to select actions."""
from .agent import PokerAgent
from .model_utils import featurize_state
import torch

ACTION_IDX_TO_NAME = {0: 'fold', 1: 'call', 2: 'raise', 3: 'check'}


class ModelAgent(PokerAgent):
    def __init__(self, name=None, starting_chips=1000, model_path=None, device=None):
        super().__init__(name=name, starting_chips=starting_chips)
        self.model_path = model_path
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        if model_path:
            self.load_model(model_path)

    def load_model(self, path):
        # Try to load the checkpoint. Newer PyTorch versions default to "weights_only=True",
        # which rejects pickled model objects. Try allowing pickled objects explicitly, then
        # fall back to a safer load attempt.
        try:
            # Prefer allowing pickled classes if the file contains a full model object.
            self.model = torch.load(path, map_location=self.device, weights_only=False)
            self.model.eval()
            return
        except TypeError:
            # Older torch versions may not accept weights_only kwarg; try without it.
            pass
        except Exception:
            # try a more permissive load (may be unsafe if file is untrusted)
            try:
                self.model = torch.load(path, map_location=self.device)
                self.model.eval()
                return
            except Exception as e:
                print(f"Failed to load model (pickled) from {path}: {e}")

        # If the file contains only a state_dict, attempt to load into a TinyMLP with inferred input size.
        try:
            state = torch.load(path, map_location=self.device)
        except Exception:
            state = None

        if not isinstance(state, dict):
            # try an adjacent .state_dict.pt file (created by the trainer)
            try:
                sd_path = path + ".state_dict.pt"
                state = torch.load(sd_path, map_location=self.device)
            except Exception:
                state = None

        if isinstance(state, dict):
            # attempt to infer input size from first linear layer weight shape
            input_size = 123
            for k, v in state.items():
                if k.endswith(".weight") and hasattr(v, "ndim") and v.ndim == 2:
                    input_size = v.shape[1]
                    break
            from training.train_model import TinyMLP
            model = TinyMLP(input_size)
            model.load_state_dict(state)
            model.to(self.device)
            model.eval()
            self.model = model
            return

        # no compatible model/state found
        print(f"Failed to load model from {path}: no compatible checkpoint found")
        self.model = None

    def make_decision(self, board, pot_size, current_bet_to_call, min_raise):
        # If no model, fallback to base behavior
        if self.model is None:
            return super().make_decision(board, pot_size, current_bet_to_call, min_raise)

        # featurize state (returns torch.FloatTensor)
        x = featurize_state(self, self.__class__.game_agents, board, pot_size, current_bet_to_call, min_raise)
        x = x.to(self.device).unsqueeze(0)

        with torch.no_grad():
            logits = self.model(x)
            action_idx = int(logits.argmax(dim=1).cpu().item())

        action = ACTION_IDX_TO_NAME.get(action_idx, 'fold')

        # Choose amounts
        if action == 'fold':
            return ('fold', 0)
        if action == 'call':
            return ('call', current_bet_to_call)
        if action == 'raise':
            # simple raise: min_raise (could be improved)
            return ('raise', min_raise)
        if action == 'check':
            return ('check', 0)

        return ('fold', 0)
