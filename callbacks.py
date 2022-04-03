import torch
from agent_code.nn_agent.genetic_algo import NeuralNetwork, GeneticAlgorithm

def setup(self):
    self.Agent = GeneticAlgorithm(500)
    self.Agent.Train = False
    self.Agent.FittestModel = NeuralNetwork().to("cpu").float()
    self.Agent.FittestModel.load_state_dict(torch.load("model/GenAlg.obj"))

def act(self, game_state: dict):
    return self.Agent.Predict(game_state)
