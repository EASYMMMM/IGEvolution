import isaacgym
from rl_games.torch_runner import Runner
from rl_games.algos_torch import model_builder
from .srl_continuous import SRLAgent
from .srl_gym import SRLGym
from .srl_players import SRLPlayerContinuous
from .srl_models import ModelSRLContinuous
from .srl_network_builder import HumanoidBuilder, SRLBuilder
from rl_games.common import env_configurations, vecenv

class SRLGym_Runner(Runner):
    def __init__(self, algo_observer=None):
        super().__init__(algo_observer)
        self.algo_factory.register_builder('srl_continuous', lambda **kwargs : SRLAgent(**kwargs))
        self.algo_factory.register_builder('srl_gym', lambda **kwargs : SRLGym(**kwargs))
        self.player_factory.register_builder('srl_continuous', lambda **kwargs : SRLPlayerContinuous(**kwargs))
        model_builder.register_model('continuous_srl', lambda network, **kwargs : ModelSRLContinuous(network))
        model_builder.register_network('amp_humanoid', lambda **kwargs : HumanoidBuilder())
        model_builder.register_network('srl', lambda **kwargs : SRLBuilder())