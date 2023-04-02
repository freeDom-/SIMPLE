
import gym
import numpy as np
import re

import config

from stable_baselines3.common import logger as sb_logger

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

ROWS = 7
COLS = 7
GRID_SIZE = COLS * ROWS
GRID_SHAPE = (COLS, ROWS)
CORNER_SQUARES = [0, ROWS-1, (COLS-1)*ROWS, COLS*ROWS-1]
CENTER_SQUARE = GRID_SIZE // 2
SIDE_SQUARES = [x for x in range(GRID_SIZE) if x // COLS in [0, ROWS-1] or x % COLS in [0, COLS-1]]
BOARDS_STORED = 2
LAST_BOARDS_STORED = BOARDS_STORED - 1
CHECK_REPETITIONS = False
STORE_TURN_COUNTER = False

ACTIONS_PER_TOKEN = COLS + ROWS
MAX_TURN_COUNT = 64

RULE_HARD_KING_CAPTURE = False
RULE_SURROUND_WHITE_WINS = False


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

    def __init__(self, verbose = False, manual = False, logger = None):
        super(TaflEnv, self).__init__()
        self.name = 'tafl'
        self.manual = manual
        self.verbose = verbose
        if logger is None:
            self.logger = sb_logger.make_output_format('stdout', config.LOGDIR, log_suffix='')
        else:
            self.logger = logger

        self.n_players = 2

        self.turns_taken = 0
        self.board = []
        self.repetition = 0
        self.last_boards = []
        self.last_repetitions = []
        self.all_tokens = []
        self.king = None
        self.black_player = None
        self.white_player = None

        self.initial_board = SMALL_BOARD
        # Actions: all board positions * maximal possible moves (+ resign)
        self.action_space = gym.spaces.Discrete(GRID_SIZE * ACTIONS_PER_TOKEN)
        # Observation: current state (black tokens, white tokens, king, 1x repeated, 2x repeated) x number of boards + current player + turn count
        # Observation: current state (black tokens, white tokens, king) x number of boards + current player
        feature_count = (5 if CHECK_REPETITIONS else 3) * BOARDS_STORED + (2 if STORE_TURN_COUNTER else 1)
        self.observation_space = gym.spaces.Box(0, 1, GRID_SHAPE + (feature_count,))

    @property
    def observation(self):
        out = []

        # Remove multiple boards to simplify problem
        all_boards = [self.board]
        all_boards.extend(self.last_boards)

        all_repetitions = [self.repetition]
        all_repetitions.extend(self.last_repetitions)

        for board, repetition in list(zip(all_boards, all_repetitions)):
            black_tokens = np.array([1 if x and x.type == BLACK_TOKEN else 0 for x in board]).reshape(GRID_SHAPE)
            white_tokens = np.array([1 if x and x.type == WHITE_TOKEN else 0 for x in board]).reshape(GRID_SHAPE)
            king = np.array([1 if x and x.type == KING else 0 for x in board]).reshape(GRID_SHAPE)
            
            out.append(black_tokens)
            out.append(white_tokens)
            out.append(king)

            if CHECK_REPETITIONS:
                if repetition == 0:
                    repeated_once = np.zeros(GRID_SHAPE)
                    repeated_twice = np.zeros(GRID_SHAPE)
                elif repetition == 1:
                    repeated_once = np.ones(GRID_SHAPE)
                    repeated_twice = np.zeros(GRID_SHAPE)
                else:
                    repeated_once = np.ones(GRID_SHAPE)
                    repeated_twice = np.ones(GRID_SHAPE)

                out.append(repeated_once)
                out.append(repeated_twice)

        # TODO: add a layer with hostile pieces (corner & throne)
        # TODO: add a layer with corners (escape fields)        

        if self.current_player == self.black_player:
            player = np.zeros(GRID_SHAPE)
        else:
            player = np.ones(GRID_SHAPE)
        out.append(player)

        if STORE_TURN_COUNTER:
            turns_taken = np.full(GRID_SHAPE, self.turns_taken / MAX_TURN_COUNT)
            out.append(turns_taken)

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
            for action in range(ACTIONS_PER_TOKEN):
                action_num = token.position * ACTIONS_PER_TOKEN + action
                legal = self.is_legal(action_num)
                legal_actions[action_num] = legal
        return legal_actions

    def action_masks(self):
        return np.array(self.legal_actions, dtype=bool)

    def parse_action(self, action):
        if type(action) is not str:
            raise TypeError("action is not of type string")

        prepared_action = action.strip().lower()
        # TODO: change re to split by any character but alphanumeric
        action_split = re.split("[, \->]+", prepared_action)
        if ROWS < 10:
            if (len(action_split) != 2 or
                not action_split[0][0].isalpha() or not action_split[0][1].isdecimal() or
                not action_split[1][0].isalpha() or not action_split[1][1].isdecimal()):
                    raise ValueError("action has an invalid format")
            start = action_split[0]
            end = action_split[1]
            start_pos = abs(int(start[1]) - ROWS) * ROWS + ord(start[0]) - ord('a')
            #end_pos = abs(int(end[1]) - ROWS) * ROWS + ord(end[0]) - ord('a')
            
            if start[1] == end[1]:
                token_action = ord(end[0]) - ord('a')
            else:
                token_action = abs(int(end[1]) - ROWS) + COLS

            action_num = start_pos * ACTIONS_PER_TOKEN + token_action
            return action_num
        else:
            raise NotImplementedError("parse_action not implemented yet for ROWS > 9")

    def parse_action_num(self, action_num):
        start, end = self.get_move(action_num)

        if ROWS < 10:
            start_x = chr(ord('a') + start % COLS)
            start_y = ROWS - start // COLS
            end_x = chr(ord('a') + end % COLS)
            end_y = ROWS - end // COLS

            move_str = f'{start_x}{start_y} {end_x}{end_y}'
            return move_str
        else:
            raise NotImplementedError("parse_action_num not implemented yet for ROWS > 9")


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
        start = action_num // ACTIONS_PER_TOKEN
        token_action = action_num % ACTIONS_PER_TOKEN
        # first half of actions represents horizontal movement, second half represents vertical movement
        if token_action < COLS:
            end = (start // COLS) * COLS + (token_action)
        else:
            end = (start % COLS) + (token_action - COLS) * COLS
        return start, end

    def is_square_player(self, board, square, player):
        if board[square] is None:
            return False
        return player.color == board[square].color

    def is_square_hostile(self, board, square, player):
        if square is None:
            return False
        return ((board[square] is not None and player.color != board[square].color) or
            square in CORNER_SQUARES or
            (square == CENTER_SQUARE and (player == self.black_player or board[CENTER_SQUARE] == None)))

    def is_path_free(self, start, end):
        if start == end:
            return False
        elif start // COLS == end // COLS:
            # Left / Right
            step = 1 if end < start else -1
        else:
            # Top / Bottom
            step = COLS if end < start else -COLS
        for x in range(end, start, step):
            if self.board[x] is not None:
                return False
        return True

    def check_game_over(self, board = None, last_board = None, player = None):
        if board is None:
            board = self.board
        if last_board is None:
            last_board = self.last_boards[0]
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
            if self.is_king_captured(board, last_board):
                return 1, True
            if RULE_SURROUND_WHITE_WINS and self.is_white_surrounded(board):
                return 1, True

        # Perpetual repetition
        if CHECK_REPETITIONS and self.repetition == 2:
            return -1, True

        if self.turns_taken > MAX_TURN_COUNT:
            return 0, True
        return 0, False #-0.01 here to encourage choosing the win?

    def is_king_captured(self, board, last_board):
        if RULE_HARD_KING_CAPTURE:
            return self.is_king_hard_captured(board)
        else:
            if self.king.position == CENTER_SQUARE:
                return self.is_king_hard_captured(board)
            else:
                neighbours = self.get_neighbour_positions(self.king.position)
                left = neighbours.get(LEFT)
                right = neighbours.get(RIGHT)
                top = neighbours.get(TOP)
                bottom = neighbours.get(BOTTOM)

                if ((self.is_square_hostile(board, left, self.white_player) and
                    self.is_square_hostile(board, right, self.white_player) and
                    (board[left] != last_board[left] or board[right] != last_board[right])) or
                    (self.is_square_hostile(board, top, self.white_player) and
                    self.is_square_hostile(board, bottom, self.white_player) and
                    (board[bottom] != last_board[bottom] or board[top] != last_board[top]))):
                    return True
                return False

    def is_king_hard_captured(self, board):
        if self.king.position in SIDE_SQUARES:
            return False
        neighbours = self.get_neighbour_positions(self.king.position)
        for direction in neighbours:
            pos = neighbours[direction]
            if not self.is_square_hostile(board, pos, self.white_player):
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
            neighbours = self.get_neighbour_positions(position)
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

    def get_neighbour_positions(self, position):
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
        neighbours = self.get_neighbour_positions(position)
        neighbour = neighbours.get(direction)
        if self.is_square_hostile(board, neighbour, self.other_player):
            return True
        return False

    def get_captures(self, board, position):
        captures = []
        
        neighbours = self.get_neighbour_positions(position)
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
            for i in range(LAST_BOARDS_STORED-1, 0, -1):
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

            if CHECK_REPETITIONS and self.board == self.last_boards[3] and self.last_boards[0] == self.last_boards[4]:
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
        self.last_boards = [[0]*GRID_SIZE]*LAST_BOARDS_STORED
        self.last_repetitions = [0]*LAST_BOARDS_STORED
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
        self.logger.debug(f'\n\n---- NEW GAME ----')
        return self.observation

    def render(self, mode='human', close=False):
        self.logger.debug('')
        if close:
            return
        if self.done:
            self.logger.debug(f'GAME OVER')
        else:
            self.logger.debug(f"It is {self.current_player.color}'s turn to move")
        for i in range(0, GRID_SIZE, COLS):
            self.logger.debug(str(COLS - i//COLS) + ' ' + ' '.join([self.get_token_representation(x) for x in self.board[i:(i+COLS)]]))
        self.logger.debug('  ' + ' '.join([chr(i) for i in range(ord('a'), ord('a') + ROWS)]))
        if self.verbose:
            self.logger.debug(f'\nObservation: \n{self.observation}')
        if not self.done:
            # TODO: parse legal_actions to readable form (do also for agents recommendations)
            self.logger.debug(f'\nLegal actions: {[i for i,o in enumerate(self.legal_actions) if o != 0]}')


    def rules_move(self):
        raise Exception('Rules based agent is not yet implemented for Tafl!')
