import events as e
import matplotlib.pyplot as plt
import torch
from agent_code.nn_agent.genetic_algo import NeuralNetwork, GeneticAlgorithm

MAX_EVAL_GAMES = 50
MAX_GAMES_FOR_EVAL = 500

def setup_training(self):
    self.Agent.Train = True

    self.NoOfEpisodesPlotX = []
    self.AverageScoreY = []
    
    self.Score = 0
    self.Episode = 0
    self.EvalEpisode = 0
    self.Eval = False

def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: list):
    reward = self.Agent.CalculateScoreAfterEvent(events)
    if self.Eval:
        self.Score += reward
    else:
        self.Agent.Scores[self.Agent.Active] += reward

def end_of_round(self, last_game_state: dict, last_action: str, events: list):
    self.Episode += 1
    
    reward = self.Agent.CalculateScoreAfterEvent(events)
    if self.Eval:
        self.Score += reward
        self.EvalEpisode += 1
        if self.EvalEpisode >= MAX_EVAL_GAMES:
            self.NoOfEpisodesPlotX.append(self.Episode)
            self.AverageScoreY.append(self.Score/MAX_EVAL_GAMES)
            self.EvalEpisode = 0
            self.Eval = False
            self.Score = 0
            self.Agent.Train = True
            plt.plot(self.NoOfEpisodesPlotX, self.AverageScoreY)
            plt.savefig("avg_score.png")
            plt.close()
    else:
        self.Agent.Scores[self.Agent.Active] += reward
        self.Agent.NextIndividual()

    if self.Episode % MAX_GAMES_FOR_EVAL == 0:
        self.Eval = True
        self.Agent.Train = False