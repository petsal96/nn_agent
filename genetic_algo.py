import numpy as np
from numpy import random as rd
import torch
from torch import nn
import random
import events as e
import pickle
import os.path

MUTATION_PROB = 1.0
MUTATION_RATIO = 0.05
PARENT_RATIO = 0.01
ELITE_RATIO = 1.0
ACTIONS = ["UP", "DOWN", "LEFT", "RIGHT", "WAIT", "BOMB"]

DEVICE = "cpu"

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.Layers = nn.Sequential(
            nn.Linear(8, 128),
            nn.ReLU(),
            nn.Linear(128, 6),
        )
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = torch.from_numpy(x).float()
        logits = self.Layers(x)
        return logits

    def predict(self, x):
        logits = self.forward(x)
        probs = nn.Softmax(dim=0)(logits).numpy()
        pred = rd.choice(6, p=probs)
        return pred

class GeneticAlgorithm:
    
    # constructor:
    def __init__(self, pop_size: int):

        # else intialize empty object:
        self.PopSize = pop_size
        self.Train = True
        self.Population = [NeuralNetwork().to(DEVICE).float() for i in range(pop_size)]
        self.Scores = [0 for i in range(pop_size)]
        self.Active = 0
        self.Generation = 0
        self.TotalCoins = 0
        self.TotalDeaths = 0
        self.TotalCrates = 0

    # feature calculation:
    def CalculateFeatureVector(self, game_state: dict):
        
        # useful vars:
        agent_pos = np.array(game_state["self"][3])
        field = np.array(game_state["field"])
        explosion = np.array(game_state["explosion_map"])
        if len(game_state["coins"]) > 0:
            coins = np.array(game_state["coins"])
        else:
            coins = np.empty((0, 2))
        if len(game_state["bombs"]) > 0:
            bombs = np.array([list(bomb[0]) for bomb in game_state["bombs"]])
        else:
            bombs = np.empty((0, 2))

        # define feature vector:
        features = np.empty(8)

        # movement (up/left/down/right/bomb):
        features[0] = int(field[agent_pos[0], agent_pos[1]-1] == 0 and explosion[agent_pos[0], agent_pos[1]-1] == 0)
        features[1] = int(field[agent_pos[0]-1, agent_pos[1]] == 0 and explosion[agent_pos[0]-1, agent_pos[1]] == 0)
        features[2] = int(field[agent_pos[0], agent_pos[1]+1] == 0 and explosion[agent_pos[0], agent_pos[1]+1] == 0)
        features[3] = int(field[agent_pos[0]+1, agent_pos[1]] == 0 and explosion[agent_pos[0]+1, agent_pos[1]] == 0)

        # next coin (up/left/down/right):
        features[4:8] = 0
        if len(coins) > 0:
            nearest_coin = coins[np.argmin(np.linalg.norm(coins-agent_pos, axis=1))]
            features[4] = agent_pos[1]-nearest_coin[1] if agent_pos[1] > nearest_coin[1] else 0
            features[5] = agent_pos[0]-nearest_coin[0] if agent_pos[0] > nearest_coin[0] else 0
            features[6] = nearest_coin[1]-agent_pos[1] if agent_pos[1] < nearest_coin[1] else 0
            features[7] = nearest_coin[0]-agent_pos[0] if agent_pos[0] < nearest_coin[0] else 0


        return features

    # cross-over:
    def CrossOver(self, first_parent, second_parent):

        # define child:
        child = NeuralNetwork().to(DEVICE).float()

        # mix layers randomly:
        for i in range(2):
            number_of_nodes = int(child.Layers[2*i].bias.shape[0])
            random_mask = random.choices([True, False], k=number_of_nodes)
            child.Layers[2*i].weight[random_mask] = first_parent.Layers[2*i].weight[random_mask]
            child.Layers[2*i].bias[random_mask] = first_parent.Layers[2*i].bias[random_mask]
            random_mask = list(~np.array(random_mask))
            child.Layers[2*i].weight[random_mask] = second_parent.Layers[2*i].weight[random_mask]
            child.Layers[2*i].bias[random_mask] = second_parent.Layers[2*i].bias[random_mask]
        

        return child

    # mutation:
    def Mutate(self, child):
        
        # mutate each layer with mutation ratio:
        for i in range(2):
            number_of_inputs = int(child.Layers[2*i].weight.shape[1]) 
            number_of_nodes = int(child.Layers[2*i].bias.shape[0])
            for j in range(number_of_nodes):
                for k in range(number_of_inputs):
                    if random.random() < MUTATION_RATIO:
                        child.Layers[2*i].weight[j, k] += random.uniform(-1, 1)
                if random.random() < MUTATION_RATIO:
                    child.Layers[2*i].bias[j] += random.uniform(-1, 1)

    # print params (for debugging)_
    def PrintParams(self, neural_net):
        for param in neural_net.parameters():
            print(param)

    # calculate score after event:
    def CalculateScoreAfterEvent(self, events: list):

        # punish or reward:
        reward = 0
        if e.COIN_COLLECTED in events:
            reward += 1
        return reward

    # prediction function:
    def Predict(self, game_state: dict):

        # calculate feature vector:
        features = self.CalculateFeatureVector(game_state)

        # predict index:
        if self.Train:
            pred = self.Population[self.Active].predict(features)
        else:
            pred = self.FittestModel.predict(features)

        return ACTIONS[pred]

    # next generation:
    def NextGeneration(self):

        # save:
        self.SaveBest()

        # print output:
        worst_score = self.Scores[np.argsort(self.Scores)[0]]
        best_score = self.Scores[np.argsort(self.Scores)[::-1][0]]
        print("\n")
        print("\n")
        print("---------------------------------------------------------")
        print("GENERATION: {g}\n".format(g=self.Generation))
        print("Total deaths: {d}, Total coins: {c}, Total crates: {cr}, Worst score: {ws}, Best score: {bs}".format(d=self.TotalDeaths, c=self.TotalCoins, cr=self.TotalCrates, ws=worst_score, bs=best_score))
        print("---------------------------------------------------------")
        print("\n")

        # get parents:
        parents = list(np.array(self.Population)[np.argsort(self.Scores)[::-1][0:int(self.PopSize*PARENT_RATIO)]])

        # get elite:
        elite = parents[0:int(len(parents)*ELITE_RATIO)]

        # reset population:
        self.Population = []
        self.Scores = [0 for i in range(self.PopSize)]
        self.Active = 0
        self.Generation += 1
        self.TotalCoins = 0
        self.TotalDeaths = 0
        self.TotalCrates = 0

        # create new population:
        self.Population = elite
        i = len(self.Population)
        while i < self.PopSize:
            first_parent, second_parent = random.sample(parents, 2)
            child = self.CrossOver(first_parent, second_parent)
            if random.random() < MUTATION_PROB:
                self.Mutate(child)
            self.Population.append(child)
            i += 1
        
    # next individual:
    def NextIndividual(self):

        # increment active and check if next generation should be created:
        self.Active += 1
        self.Rounds = 0
        if self.Active >= self.PopSize:
            self.NextGeneration()

    # save model:
    def SaveBest(self):
        
        # save fittest:
        self.FittestModel = self.Population[np.argmax(self.Scores)]
        torch.save(self.Population[np.argmax(self.Scores)].state_dict(), "model/GenAlg.obj")