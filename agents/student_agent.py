from typing import Callable, List

from commons import CommonsAgent, CommonsPerception
from communication import AgentAction


class StudentAgent(CommonsAgent):
    def __init__(self, agent_id):
        super(StudentAgent, self).__init__(agent_id)
        self.last_perception = None

    @staticmethod
    def _fair_share(num_agents: int) -> float:
        if num_agents <= 0:
            return 0.0
        return 1.0 / (2.0 * num_agents)

    def specify_share(self, perception: CommonsPerception) -> float:
        return self._fair_share(perception.num_agents)

    def negotiation_response(self, negotiation_round: int, perception: CommonsPerception,
                             utility_func: Callable[[float, float, List[float]], float]) -> AgentAction:
        target_share = self._fair_share(perception.num_agents)
        current_shares = perception.resource_shares or {self.id: target_share}
        current_share = current_shares.get(self.id, target_share)

        if self._is_close_to_target(current_shares, target_share):
            return AgentAction(self.id, resource_share=current_share, no_action=True)

        if perception.aggregate_adjustment:
            adjusted_shares = self._apply_adjustment(current_shares, perception.aggregate_adjustment)
            if self._is_valid(adjusted_shares) and self._is_close_to_target(adjusted_shares, target_share):
                return AgentAction(self.id, resource_share=current_share, no_action=True)

        consumption_adjustment = {
            agent_id: target_share - share
            for agent_id, share in current_shares.items()
        }

        adjusted_shares = self._apply_adjustment(current_shares, consumption_adjustment)
        if not self._is_valid(adjusted_shares):
            return AgentAction(self.id, resource_share=current_share, no_action=True)

        return AgentAction(
            self.id,
            resource_share=current_share,
            consumption_adjustment=consumption_adjustment,
            no_action=False
        )

    def inform_round_finished(self, negotiation_round: int, perception: CommonsPerception):
        ## information sent to the agent once the current round (including all adjustment rounds) is finished
        self.last_perception = perception

    @staticmethod
    def _apply_adjustment(shares, adjustment):
        return {
            agent_id: share + adjustment.get(agent_id, 0.0)
            for agent_id, share in shares.items()
        }

    @staticmethod
    def _is_valid(shares) -> bool:
        return all(share >= 0.0 for share in shares.values()) and sum(shares.values()) < 1.0

    @staticmethod
    def _is_close_to_target(shares, target_share: float) -> bool:
        return all(abs(share - target_share) < 1e-9 for share in shares.values())

