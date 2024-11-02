"""
Tic-Tac-Toe with AI

This program is a Tic-Tac-Toe game using classes and objects, with AI opponent support.
"""

from enum import Enum, auto
import numpy as np
from sklearn.tree import DecisionTreeClassifier
import random


class Cell(Enum):
    X = auto()
    O = auto()
    EMPTY = auto()


class Player(Enum):
    X = auto()
    O = auto()


class Board:
    """Tic-Tac-Toe board."""

    BOARD_SIZE: int = 3  # Game constants

    def __init__(self):
        """Initializes the board with empty cells"""
        self.board = [
            [Cell.EMPTY for _ in range(self.BOARD_SIZE)]
            for _ in range(self.BOARD_SIZE)
        ]

    def printBoard(self):
        """Print the game board"""
        header: str = (
            "|R\\C|"
            + " |".join(f" {i}" for i in range(self.BOARD_SIZE))
            + " |"
        )
        separator: str = "-" * len(header)

        print(separator)
        print(header)
        print(separator)

        for row in range(self.BOARD_SIZE):
            row_str: str = f"| {row} |"
            for col in range(self.BOARD_SIZE):
                if self.board[row][col] == Cell.X:
                    row_str += " X |"
                elif self.board[row][col] == Cell.O:
                    row_str += " O |"
                else:
                    row_str += "   |"
            print(row_str)
            print(separator)
        print()

    def get_numpy_board(self):
        """Convert the board to numpy array format for AI processing"""
        result = np.empty((3, 3), dtype=str)
        for i in range(3):
            for j in range(3):
                if self.board[i][j] == Cell.X:
                    result[i][j] = "X"
                elif self.board[i][j] == Cell.O:
                    result[i][j] = "O"
                else:
                    result[i][j] = " "
        return result


class Game:
    """Tic-Tac-Toe game."""

    def __init__(self, use_ai=False):
        """Initializes the board and the turn of player."""
        self.board = Board()
        self.turn = Player.X
        self.use_ai = use_ai
        if use_ai:
            self.ai = TicTacToeAI()

    def switchPlayer(self):
        """Switch player."""
        if self.turn == Player.X:
            self.turn = Player.O
        else:
            self.turn = Player.X

    def validateEntry(self, row: int, col: int) -> bool:
        """Valid the user's entry."""
        board_size = self.board.BOARD_SIZE
        board = self.board.board
        if not (0 <= row < board_size and 0 <= col < board_size):
            print(
                "Invalid entry: try again.\nRow & column numbers must be either 0, 1, or 2.\n"
            )
            return False

        if board[row][col] != Cell.EMPTY:
            print(
                "That cell is already taken.\nPlease make another selection.\n"
            )
            return False
        return True

    def checkFull(self):
        """Check if the board is full."""
        return all(
            cell != Cell.EMPTY for row in self.board.board for cell in row
        )

    def checkWin(self) -> bool:
        """Check if the current player has won."""
        board = self.board.board
        board_size = self.board.BOARD_SIZE

        # Check rows
        for row in board:
            if row[0] != Cell.EMPTY and all(cell == row[0] for cell in row):
                return True

        # Check columns
        for col in zip(*board):
            if col[0] != Cell.EMPTY and all(cell == col[0] for cell in col):
                return True

        # Check main diagonal
        if board[0][0] != Cell.EMPTY and all(
            board[i][i] == board[0][0] for i in range(board_size)
        ):
            return True

        # Check anti-diagonal
        if board[0][board_size - 1] != Cell.EMPTY and all(
            board[i][board_size - 1 - i] == board[0][board_size - 1]
            for i in range(board_size)
        ):
            return True

        return False

    def checkEnd(self) -> bool:
        """Check if the game is ended."""
        if self.checkWin():
            print(f"{self.turn.name} IS THE WINNER!!!")
            return True
        if self.checkFull():
            print("DRAW! NOBODY WINS!")
            return True
        return False

    def get_ai_move(self):
        """Get the AI's next move"""
        numpy_board = self.board.get_numpy_board()
        row, col = self.ai.get_move(numpy_board)
        return row, col

    def playGame(self):
        """Play a single game of Tic-Tac-Toe"""
        board = self.board.board
        print(f"\nNew Game: {self.turn.name} goes first.\n")
        self.board.printBoard()

        while True:
            print(f"{self.turn.name}'s turn.")

            if self.use_ai and self.turn == Player.O:
                # AI's turn
                row, col = self.get_ai_move()
                print(f"AI places O at row {row}, column {col}")
            else:
                # Human's turn
                print(f"Where do you want your {self.turn.name} placed?")
                try:
                    row, col = map(
                        int,
                        input(
                            "Please enter row number and column number separated by a comma.\n"
                        ).split(","),
                    )
                except ValueError:
                    print("Invalid entry: try again.\n")
                    continue

                print(f"You have entered row #{row}")
                print(f"          and column #{col}")

            if not self.validateEntry(row, col):
                continue

            board[row][col] = getattr(Cell, self.turn.name)
            self.board.printBoard()

            if self.checkEnd():
                return

            self.switchPlayer()


class TicTacToeAI:
    def __init__(self, filename="tictac_final.txt"):
        self.model = DecisionTreeClassifier()
        self.filename = filename
        self.X, self.y = self.load_training_data()
        self.train()

    def load_training_data(self):
        """Read datasets from file."""
        board_states = []
        moves = []
        try:
            with open(self.filename, "r") as file:
                for line in file:
                    # Parse each line of data
                    data = list(map(int, line.strip().split()))
                    features = data[:9]  # First 9 are board states
                    move = data[9]  # 10th number is best move position
                    board_states.append(features)
                    moves.append(move)
            return np.array(board_states), np.array(moves)
        except FileNotFoundError:
            print(f"File {self.filename} not found.")
            return None, None

    def train(self):
        """Train model"""
        self.model.fit(self.X, self.y)
        print(f"Model training completed with {len(self.X)} training data")

    def get_move(self, board):
        """Predict next move, chooses randomly if occupied"""
        features = self.board_to_features(board)
        move = self.model.predict([features])[0]

        if (
            board.flatten()[move] != " "
        ):  # If occupied, randomly choose empty cell
            empty_cells = [
                i for i, cell in enumerate(board.flatten()) if cell == " "
            ]
            if empty_cells:
                move = random.choice(empty_cells)
            else:
                raise ValueError("No moves available: the board is full.")

        return move // 3, move % 3

    def board_to_features(self, board):
        """Convert board to feature vector"""
        return [
            1 if cell == "X" else -1 if cell == "O" else 0
            for cell in board.flatten()
        ]


def main():
    """Main game loop"""
    print("Welcome to Tic-Tac-Toe!")
    while True:
        mode = input("Play against AI? (y/n): ").lower()
        game = Game(use_ai=(mode == "y"))
        game.playGame()

        again = input("Another game? Enter Y or y for yes.\n")
        if again.lower() != "y":
            break

    print("Thanks for playing!")


if __name__ == "__main__":
    main()
