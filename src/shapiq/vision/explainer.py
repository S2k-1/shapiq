from typing_extensions import Literal

from matplotlib.pylab import TYPE_CHECKING

from typing import TYPE_CHECKING, Any

from shapiq.interaction_values import InteractionValues
from shapiq.game_theory.indices import is_empty_value_the_baseline

from .imputer import ImageImputer
from .players import PlayerStrategy, SuperpixelStrategy, PatchStrategy
from .masking import MaskingStrategy, MeanColorMasking

from shapiq.explainer.base import Explainer
from shapiq.explainer.configuration import setup_approximator
from shapiq.explainer.custom_types import ExplainerIndices

if TYPE_CHECKING:
    from shapiq.typing import Model
    

import numpy as np

ImageExplainerIndices = ExplainerIndices
ModelType = Literal["resnet", "vit"]

class ImageExplainer(Explainer):
    """Explainer for vision models driven by ImageImputer class.

    Treats the (image, model, imputer) triple as a cooperative ``Game`` and delegates to a
    shapiq approximator. The user-supplied ImageImputer class is the value function: it
    masks the image according to a coalition, calls the model, and returns the payoff.
    """

    def __init__(
        self,
        model,
        data: np.ndarray | None = None,
        *,
        model_type: ModelType,
        player_strategy: PlayerStrategy | None = None,
        masking_strategy: MaskingStrategy | None = None,
        class_index: int | None = None,
        index: ImageExplainerIndices = "k-SII",
        max_order: int = 2,
        random_state: int | None = None,
        **kwargs: Any,
    ) -> None:
        
        super().__init__(model=model, index=index, max_order=max_order)
        
        self._model_type = model_type

        player_strategy = player_strategy or self._default_player_strategy()
        masking_strategy = masking_strategy or self._default_masking_strategy()
        
        self._imputer = ImageImputer(
            model=model, 
            image=data, 
            masking_strategy=masking_strategy, 
            player_strategy=player_strategy, 
            model_type=model_type
        )
        
        self._approximator = setup_approximator(
            approximator="auto",
            index=index,
            max_order=self.max_order,
            n_players=self._imputer.n_players,
            random_state=random_state,
        )
        

    def explain_function(
        self, x:np.ndarray | None, *, budget: int = 64
    ) -> InteractionValues:
        
        interaction_values = self.approximator.approximate(budget=budget, game=self.imputer)
        interaction_values.baseline_value = self.baseline_value
        
        # Adjust the Baseline Value if the empty value is the baseline
        if is_empty_value_the_baseline(interaction_values.index):
            interaction_values[()] = interaction_values.baseline_value
        
        return interaction_values
    
    def _default_player_strategy(self) -> PlayerStrategy:
        if self._model_type == "vit":
            return PatchStrategy(grid_size=12, patch_size=9)
        
        elif self._model_type == "resnet":
            return SuperpixelStrategy(n_segments=10)
        
        else:
            raise ValueError(f"Unsupported model type: {self._model_type}")
        
    def _default_masking_strategy(self) -> MaskingStrategy:
        return MeanColorMasking()

    @property
    def baseline_value(self) -> float:
        """Returns the baseline value of the explainer."""
        return self.imputer.empty_prediction