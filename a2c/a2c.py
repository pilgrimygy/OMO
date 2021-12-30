import os
import torch
import torch.nn.functional as F
from torch.optim import Adam
from sac.model import GaussianPolicy, ValueNetwork



class A2C(object):
    def __init__(self, num_inputs, action_space, args) -> None:
        torch.autograd.set_detect_anomaly(True)
        self.gamma = args['gamma']
        self.value_loss_coef = args["value_loss_coef"]
        self.entropy_coef = args["entropy_coef"]

        self.device = torch.device("cuda")

        self.critic = ValueNetwork(num_inputs, args["hidden_size"]).to(device=self.device)

        self.critic_optim = Adam(self.critic.parameters(), lr=args["critic_lr"])

        self.policy = GaussianPolicy(num_inputs, action_space.shape[0], args["hidden_size"], action_space).to(self.device)

        self.policy_optim = Adam(self.policy.parameters(), lr=args["policy_lr"])
    
    def select_action(self, state, eval=False):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        if eval == False:
            action, _, _ = self.policy.sample(state)
        else:
            _, _, action = self.policy.sample(state)
        return action.detach().cpu().numpy()[0]
    
    def update(self, states, actions, returns, next_states, masks, args):
        states = torch.FloatTensor(states).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device).unsqueeze(1)
        masks = torch.FloatTensor(masks).to(self.device).unsqueeze(1)

        vf_loss = F.mse_loss(returns, self.critic(states), reduction='sum') / args["trajectory_num"]

        log_prob, entropy = self.policy.evaluate_actions(states, actions)

        policy_loss = torch.sum((returns - self.critic(states).detach()) * log_prob) / args["trajectory_num"]

        loss = self.value_loss_coef * vf_loss + policy_loss - self.entropy_coef * entropy

        self.policy_optim.zero_grad()
        self.critic_optim.zero_grad()
        loss.backward()
        self.critic_optim.step()
        self.policy_optim.step()

        return vf_loss.item(), policy_loss.item(), entropy.item()