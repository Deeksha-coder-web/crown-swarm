"""Crown-shyness-inspired decentralized swarm dispersion simulator."""
from .episode import EpisodeConfig, run_episode
from .controllers import CONTROLLERS, TUNABLE, make_controller
from .floorplans import TRAIN_PLANS, TEST_PLANS, get_plan
