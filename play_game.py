from random import randint, random
import pickle
import numpy as np

class Board:

    def __init__(self):
        with open('monopoly_board.data', mode='rb') as f:
            self.board_data = pickle.loads(f.read())
        self.map_square_to_property()
        self.ownership = np.ones(22,) * -1
        self.houses = np.zeros(22,)
    
    def ownership_one_hot(self):
        self.ownership_array = [[], [], []]
        for index in range(len(self.ownership)):
            for player_index in range(len(self.ownership_array)):
                self.ownership_array[player_index].append(0)
            if self.ownership[index] != -1:
                self.ownership_array[int(self.ownership[index])][index] = 1

    def map_square_to_property(self):
        self.space_property_map = {}
        self.property_num = 0
        for space_num in range(40):
            if self.board_data[space_num]['purchaseable'] == 1:
                self.space_property_map[space_num] = self.property_num
                self.property_num += 1
        self.property_space_map = {b: a for a, b in self.space_property_map.items()}
    
class Player:

    def __init__(self, model, start_location=0, start_funds=1500):
        self.model = model
        self.location = start_location
        self.funds = start_funds

class Game:

    def __init__(self, model, epsilon):
        self.neural_model = model
        self.epsilon = epsilon
        self.ownership_inputs = []
        self.predictions = []
        self.revenues = []
        self.input_history = []
        self.output_history = []
    
    def get_observations(self):
        return np.array(self.input_history), np.array(self.output_history)

    def save_observations(self):
        self.future_revenues = []
        for time_step in range(len(self.revenues)):
            self.future_revenue = self.revenues[-1] - self.revenues[time_step]
            for property_id in range(len(self.future_revenue)):
                if self.future_revenue[property_id] == 0:
                    self.future_revenue[property_id] = self.predictions[time_step][property_id]
            self.future_revenues.append(self.future_revenue)
        for step in range(len(self.future_revenues)):
            self.input_history.append(self.ownership_inputs[step])
            self.output_history.append(self.future_revenues[step])
        self.ownership_inputs = []
        self.predictions = []
        self.revenues = []

    def buy_property(self, player_index):
        if self.players[player_index].funds >= self.board.board_data[self.players[player_index].location]['buy_price']:
            self.board.ownership[self.board.space_property_map[self.players[player_index].location]] = player_index
            self.players[player_index].funds -= self.board.board_data[self.players[player_index].location]['buy_price']

    def upgrade_property(self, player_index, property_index):
        if self.players[player_index].funds >= self.board.board_data[self.board.property_space_map[property_index]]['house_price']:
            self.board.houses[property_index] += 1
            self.players[player_index].funds -= self.board.board_data[self.board.property_space_map[property_index]]['house_price']

    def set_up_game(self, models):
        self.players = []
        for model in models:
            self.players.append(Player(model))
        self.board = Board()
    
    def play_games(self, num_games):
        for _ in range(num_games):
            self.simulate_game()
            #print("Finished game " + str(_) + "/" + str(num_games))
    
    def remove_player(self, player_id):
        self.players[player_id].funds = 0
        for property_num in range(len(self.board.ownership)):
            if self.board.ownership[property_num] == player_id:
                self.board.ownership[property_num] = -1
                self.board.houses[property_num] = 0
        self.bankrupt_players.append(player_id)

    def simulate_game(self):
        self.set_up_game([self.neural_model, self.neural_model, self.neural_model])
        self.bankrupt_players = []
        while len(self.bankrupt_players) < 2:
            for index in range(len(self.players)):
                self.have_go(index)
        self.save_observations()

    def have_go(self, player_index):
        if player_index not in self.bankrupt_players:
            if self.players[player_index].funds <= 0:
                self.remove_player(player_index)
            self.current_dice_roll = randint(1, 6) + randint(1, 6)
            self.players[player_index].location += self.current_dice_roll
            while self.players[player_index].location > 39:
                self.players[player_index].location -= 40
            if self.players[player_index].location in self.board.space_property_map:
                self.current_property = self.board.space_property_map[self.players[player_index].location]
                self.current_owner = self.board.ownership[self.current_property]
                self.board.ownership_one_hot()
                self.input_ownership = np.array(self.board.ownership_array).flatten().reshape(1, 66)
                self.valuation = self.players[player_index].model.predict(self.input_ownership)
                self.predictions.append(self.valuation[0])
                self.ownership_inputs.append(self.input_ownership[0])
                if len(self.revenues) == 0:
                    self.revenues.append(np.zeros(22))
                else:
                    self.revenues.append(np.copy(self.revenues[-1]))
                if self.current_owner != -1 and self.current_owner not in self.bankrupt_players:
                    self.current_rent = self.board.board_data[self.players[player_index].location]['prices'][int(self.board.houses[self.current_property])]
                    self.players[player_index].funds -= self.current_rent
                    self.players[int(self.current_owner)].funds += self.current_rent
                    self.revenues[-1][self.current_property] += self.current_rent
                else:
                    self.buy_price = self.board.board_data[self.players[player_index].location]['buy_price']
                    if random() < self.epsilon:
                        self.valuation = np.random.randint(400, size=22).reshape(1, 22)
                    if self.valuation[0][self.current_property] >= self.buy_price:
                        self.buy_property(player_index)
                    self.owned_indices = []
                    for val_index in range(22):
                        if self.board.ownership[val_index] == player_index and self.board.houses[val_index] < 5:
                            self.owned_indices.append(True)
                        else:
                            self.owned_indices.append(False)
                    if True in self.owned_indices:
                        self.upgrade_eligible = np.array(self.valuation[0])[np.array(self.owned_indices)]
                        self.upgrade_indices = np.array(range(22))[np.array(self.owned_indices)]
                        self.upgrade_property(player_index, self.upgrade_indices[np.argmax(self.upgrade_eligible)])