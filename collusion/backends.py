from .agents import MockAgent, LLMAgent
from .llm import Client
from .trusted import HeuristicMonitor, LLMMonitor, MockMediator, LLMMediator


class MockBackend:
    name = "mock"

    def agent(self, idx, model, rng):
        return MockAgent(model, rng)

    def monitor(self):
        return HeuristicMonitor()

    def mediator(self, kind, rng):
        return MockMediator(kind, rng)


class LLMBackend:
    """Agents use the model spec in the config; monitor/auditor/mediator use the trusted model."""
    name = "llm"

    def __init__(self, trusted_model, provider="anthropic"):
        self.provider = provider
        self._trusted = Client(trusted_model, provider)

    def agent(self, idx, model, rng):
        return LLMAgent(Client(model, self.provider))

    def monitor(self):
        return LLMMonitor(self._trusted)

    def mediator(self, kind, rng):
        return LLMMediator(kind, self._trusted)
