
import gym
import numpy as np

import config

from stable_baselines import logger

BLACK = "BLACK"
WHITE = "WHITE"

LEFT   = "LEFT"
RIGHT  = "RIGHT"
TOP    = "TOP"
BOTTOM = "BOTTOM"

EMPTY       = 0
BLACK_TOKEN = -1
WHITE_TOKEN = 1
KING        = 2

TYPE_TO_STRING = {
    EMPTY: '.',
    WHITE_TOKEN: 'T',
    KING: 'K',
    BLACK_TOKEN: 't'
}

ROWS = 11
COLS = 11
GRID_SIZE = COLS * ROWS
GRID_SHAPE = (COLS, ROWS)
CORNER_SQUARES = [0, ROWS-1, (COLS-1)*ROWS, COLS*ROWS-1]
CENTER_SQUARE = GRID_SIZE // 2
SIDE_SQUARES = [x for x in range(GRID_SIZE) if x // COLS in [0, ROWS-1] or x % COLS in [0, COLS-1]]

MAX_TURN_COUNT = 512


DEFAULT_BOARD = [
     0,  0,  0, -1, -1, -1, -1, -1,  0,  0,  0,
     0,  0,  0,  0,  0, -1,  0,  0,  0,  0,  0,
     0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
    -1,  0,  0,  0,  0,  1,  0,  0,  0,  0, -1,
    -1,  0,  0,  0,  1,  1,  1,  0,  0,  0, -1,
    -1, -1,  0,  1,  1,  2,  1,  1,  0, -1, -1,
    -1,  0,  0,  0,  1,  1,  1,  0,  0,  0, -1,
    -1,  0,  0,  0,  0,  1,  0,  0,  0,  0, -1,
     0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
     0,  0,  0,  0,  0, -1,  0,  0,  0,  0,  0,
     0,  0,  0, -1, -1, -1, -1, -1,  0,  0,  0
]

TEST_BOARD = [
     0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
     0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
     0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
     0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
     0,  0,  0,  0,  0, -1, -1,  0,  0,  0,  0,
     0,  0,  0,  0, -1,  0,  2, -1,  0,  0,  0,
     0,  0,  0,  0,  0, -1,  0, -1,  0,  0,  0,
     0,  0,  0,  0,  0, -1,  0,  0,  0,  0,  0,
     0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
     0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
     0,  0,  0,  1,  1,  1,  1,  1,  0,  0,  0
]

SMALL_BOARD = [
     0,  0,  0, -1,  0,  0,  0,
     0,  0,  0, -1,  0,  0,  0,
     0,  0,  0,  1,  0,  0,  0,
    -1, -1,  1,  2,  1, -1, -1,
     0,  0,  0,  1,  0,  0,  0,
     0,  0,  0, -1,  0,  0,  0,
     0,  0,  0, -1,  0,  0,  0
]


class Player():
    def __init__(self, id, color):
        self.id = id
        self.color = color
        self.tokens = []


class Token():
    count = 0

    def __init__(self, type, color, position):
        self.id = Token.count
        self.type = type
        self.color = color
        self.position = position
        Token.count += 1

    @property
    def x(self):
        return self.position % COLS

    @property
    def y(self):
        return self.position // COLS


class TaflEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, verbose = False, manual = False):
        super(TaflEnv, self).__init__()
        self.name = 'tafl'
        self.manual = manual
        self.verbose = verbose
        
        self.n_players = 2
        self.actions_per_token = COLS + ROWS

        self.turns_taken = 0
        self.board = []
        self.repetition = 0
        self.last_boards = []
        self.last_repetitions = []
        self.all_tokens = []
        self.king = None
        self.black_player = None
        self.white_player = None

        self.initial_board = TEST_BOARD
        # Actions: all board positions * maximal possible moves (+ resign)
        self.action_space = gym.spaces.Discrete(GRID_SIZE * self.actions_per_token)
        # Observation: current state (black tokens, white tokens, king, 1x repeated, 2x repeated) + last seven states + current player + turn count +  22 x actions
        self.observation_space = gym.spaces.Box(0, 1, GRID_SHAPE + (54,))

    @property
    def observation(self):
        out = []

        all_boards = [self.board]
        all_boards.extend(self.last_boards)

        all_repetitions = [self.repetition]
        all_repetitions.extend(self.last_repetitions)

        for board, repetition in list(zip(all_boards, all_repetitions)):
            black_tokens = np.array([1 if x and x.type == BLACK_TOKEN else 0 for x in board]).reshape(GRID_SHAPE)
            white_tokens = np.array([1 if x and x.type == WHITE_TOKEN else 0 for x in board]).reshape(GRID_SHAPE)
            king = np.array([1 if x and x.type == KING else 0 for x in board]).reshape(GRID_SHAPE)
            
            if repetition == 0:
                repeated_once = np.zeros(GRID_SHAPE)
                repeated_twice = np.zeros(GRID_SHAPE)
            elif repetition == 1:
                repeated_once = np.ones(GRID_SHAPE)
                repeated_twice = np.zeros(GRID_SHAPE)
            else:
                repeated_once = np.ones(GRID_SHAPE)
                repeated_twice = np.ones(GRID_SHAPE)

            out.append(black_tokens)
            out.append(white_tokens)
            out.append(king)
            out.append(repeated_once)
            out.append(repeated_twice)

        if self.current_player == self.black_player:
            player = np.zeros(GRID_SHAPE)
        else:
            player = np.ones(GRID_SHAPE)
        out.append(player)

        turns_taken = np.full(GRID_SHAPE, self.turns_taken / MAX_TURN_COUNT)
        out.append(turns_taken)

        la = self.legal_actions
        la.resize(22, 11, 11)
        out.extend(la)

        out = np.stack(out, axis=-1)
        return out

    @property
    def current_player(self):
        return self.players[self.current_player_num]
        
    @property
    def other_player(self):
        return self.players[(self.current_player_num + 1) % self.n_players]

    @property
    def legal_actions(self):
        legal_actions = np.zeros(self.action_space.n)
        for id, token in enumerate(self.current_player.tokens):
            for action in range(self.actions_per_token):
                action_num = token.position * self.actions_per_token + action
                legal = self.is_legal(action_num)
                legal_actions[action_num] = legal
        return legal_actions

    def get_token_representation(self, token, position = None):
        if position:
            repr = TYPE_TO_STRING.get(token.type, lambda: "Invalid id") if token else f"{position}".zfill(3)
            repr = f'{repr:^3}'
        else:
            type = token.type if token else EMPTY
            repr = TYPE_TO_STRING.get(type, lambda: "Invalid id")
        return repr

    def is_legal(self, action_num, board = None, player = None):
        if board is None:
            board = self.board
        if player is None:
            player = self.current_player
        start, end = self.get_move(action_num)
        token = board[start]
        if token is None:
            return 0
        if (token.color == player.color and self.is_path_free(start, end) and
            (token.type == KING or (end != CENTER_SQUARE and end not in CORNER_SQUARES))):
            return 1
        else:
            return 0

    def get_move(self, action_num):
        start = action_num // self.actions_per_token
        token_action = action_num % self.actions_per_token
        # 0-10 horizontal movement, 11-21 vertical movement
        if token_action < COLS:
            end = (start // COLS) * COLS + (token_action)
        else:
            end = (start % COLS) + (token_action - COLS) * COLS
        return start, end

    def is_square_player(self, board, square, player):
        if board[square] is None:
            return False
        return player.color == board[square].color

    def is_path_free(self, start, end):
        if start == end:
            return False
        elif start // COLS == end // COLS:
            if end < start:
                # Left
                step = 1
            else:
                # Right
                step = -1 
        else:
            if end < start:
                # Top
                step = COLS
            else:
                # Bottom
                step = -COLS
        for x in range(end, start, step):
            if self.board[x] is not None:
                return False
        return True

    def check_game_over(self, board = None , player = None):
        if board is None:
            board = self.board
        if player is None:
            player = self.current_player

        if self.current_player == self.white_player:
            # King escaped
            if self.king.position in CORNER_SQUARES:
                return 1, True
            if len(self.other_player.tokens) == 0:
                return 1, True
            # TODO: Exit forts
        else:
            # King captured
            if self.is_king_captured(board):
                return 1, True
            if self.is_white_surrounded(board):
                return 1, True

        # Perpetual repetition
        if self.repetition == 2:
            return -1, True

        if self.turns_taken > MAX_TURN_COUNT:
            return 0, True
        return 0, False #-0.01 here to encourage choosing the win?

    def is_king_captured(self, board):
        if self.king.position in SIDE_SQUARES:
            return False
        neighbours = self.get_neighbour_positions(board, self.king.position)
        for direction in neighbours:
            pos = neighbours[direction]
            if not self.is_square_player(board, pos, self.black_player) and pos != CENTER_SQUARE:
                return False
        return True

    def is_white_surrounded(self, board):
        white_tokens = self.white_player.tokens.copy()
        while white_tokens:
            token = white_tokens.pop()
            surrounded, tokens = self.is_token_surrounded(board, token)
            if not surrounded:
                return False
            for token in tokens:
                white_tokens.remove(token)
        return True

    def is_token_surrounded(self, board, token):
        positions = [token.position]
        visited = [token.position]
        tokens = []

        while positions:
            position = positions.pop()
            visited.append(position)
            neighbours = self.get_neighbour_positions(board, position)
            for _, position in neighbours.items():
                if position not in visited:
                    tok = board[position]
                    if position in SIDE_SQUARES:
                        return False, []
                    if tok is None:
                        positions.append(position)
                    elif tok.color == token.color:
                        positions.append(position)
                        tokens.append(tok)

        return True, tokens

    def get_neighbour_positions(self, board, position):
        row = position // COLS
        neighbours = {}
        if position-1 >= 0 and (position-1) // COLS == row:
            neighbours[LEFT] = position-1
        if position+1 < GRID_SIZE and (position+1) // COLS == row:
            neighbours[RIGHT] = position+1
        if position-COLS >= 0:
            neighbours[TOP] = position-COLS
        if position+COLS < GRID_SIZE:
            neighbours[BOTTOM] = position+COLS
        return neighbours

    def is_captured(self, board, position, direction):
        if board[position] == self.king:
            return False
        neighbours = self.get_neighbour_positions(board, position)
        neighbour = neighbours.get(direction)
        if neighbour is not None:
            if (self.is_square_player(board, neighbour, self.current_player) or neighbour in CORNER_SQUARES or
                (neighbour == CENTER_SQUARE and (board[neighbour] == None or self.current_player == self.white_player))):
                return True
        return False

    def get_captures(self, board, position):
        captures = []
        
        neighbours = self.get_neighbour_positions(board, position)
        for direction in neighbours:
            neighbour_position = neighbours[direction]
            # TODO: check shieldwall captures
            if self.is_square_player(board, neighbour_position, self.other_player) and self.is_captured(board, neighbour_position, direction):
                captures.append(neighbour_position)
        return captures

    def step(self, action):
        reward = [0,0]

        # check move legality
        if not self.is_legal(action):
            done = True
            reward = [1,1]
            reward[self.current_player_num] = -1
        else:
            for i in range(4, 0, -1):
                self.last_boards[i] = self.last_boards[i-1]
                self.last_repetitions[i] = self.last_repetitions[i-1]
            self.last_boards[0] = self.board.copy()
            self.last_repetitions[0] = self.repetition

            start, end = self.get_move(action)
            self.board[end] = self.board[start]
            self.board[start] = None
            self.board[end].position = end
            for capture in self.get_captures(self.board, end):
                token = self.board[capture]
                self.other_player.tokens.remove(token)
                self.all_tokens[token.id] = None
                self.board[capture] = None

            self.turns_taken += 1

            if self.board == self.last_boards[3] and self.last_boards[0] == self.last_boards[4]:
                self.repetition = self.last_repetitions[3] + 1
            else:
                self.repetition = 0

            r, done = self.check_game_over()
            reward = [-r,-r]
            reward[self.current_player_num] = r

        self.done = done

        if not done:
            self.current_player_num = (self.current_player_num + 1) % 2

        return self.observation, reward, done, {}

    def reset(self):
        self.board = []
        self.repetition = 0
        self.last_boards = [[0]*121, [0]*121, [0]*121, [0]*121, [0]*121]
        self.last_repetitions = [0]*5
        self.all_tokens = []
        self.king = None
        Token.count = 0
        self.players = [Player('1', BLACK), Player('2', WHITE)]
        self.black_player = self.players[0]
        self.white_player = self.players[1]

        for position, type in enumerate(self.initial_board):
            token = None
            if type != 0:
                if type < 0:
                    token = Token(type, BLACK, position)
                    self.black_player.tokens.append(token)
                elif type > 0:
                    token = Token(type, WHITE, position)
                    self.white_player.tokens.append(token)
                    if token.type == KING:
                        self.king = token
                self.all_tokens.append(token)
            self.board.append(token)
        self.current_player_num = 0
        self.turns_taken = 0
        self.done = False
        logger.debug(f'\n\n---- NEW GAME ----')
        return self.observation

    def render(self, mode='human', close=False):
        logger.debug('')
        if close:
            return
        if self.done:
            logger.debug(f'GAME OVER')
        else:
            logger.debug(f"It is {self.current_player.color}'s turn to move")
        for i in range(0, GRID_SIZE, COLS):
            logger.debug(' '.join([self.get_token_representation(x) for x in self.board[i:(i+COLS)]]))
            #logger.debug(' '.join([self.get_token_representation(x, position=pos+i) for pos, x in enumerate(self.board[i:(i+COLS)])]))
        if self.verbose:
            logger.debug(f'\nObservation: \n{self.observation}')
        if not self.done:
            #la = {}
            #for action, legal in enumerate(self.legal_actions):
            #    if legal:
            #        start, end = self.get_move(action)
            #        la[action] = f'{start}->{end}'
            logger.debug(f'\nLegal actions: {[i for i,o in enumerate(self.legal_actions) if o != 0]}')


    def rules_move(self):
        # TODO: check if this method is still correct
        WRONG_MOVE_PROB = 0.00001
        player = self.current_player_num

        legal_actions = []
        # Check win moves
        for action in range(self.action_space.n):
            if self.is_legal(action):
                legal_actions.append(action)
                new_board = self.board.copy()
                start, end = self.get_move(action)
                new_board[end] = new_board[start]
                new_board[start] = None
                new_board[end].position = end
                for capture in self.get_captures(new_board, end):
                    new_board[capture] = None
                _, done = self.check_game_over(new_board, player)
                new_board[end].position = start
                if done:
                    action_probs = [WRONG_MOVE_PROB] * self.action_space.n
                    action_probs[action] = 1 - WRONG_MOVE_PROB * (self.action_space.n - 1)
                    return action_probs

        # For every legal action evenly distribute probabilities
        if legal_actions:
            prob = 1/len(legal_actions)
            masked_action_probs = [0] * self.action_space.n
            for action in legal_actions:
                masked_action_probs[action] = prob
        else:
            prob = 1/self.action_space.n
            masked_action_probs = [prob] * self.action_space.n

        return masked_action_probs

