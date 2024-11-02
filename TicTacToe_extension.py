"""
Tic-Tac-Toe

This program is a simple Tic-Tac-Toe game using classes and objects.
    
    
"""

from enum import Enum, auto
from typing import List, Tuple


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
        print()  # Make sure there is a newline


class Game:
    """Tic-Tac-Toe game."""

    def __init__(self):
        """Initializes the board and the turn of player."""
        self.board = Board()
        self.turn = Player.X

    def switchPlayer(self):
        """Switch player."""
        if self.turn == Player.X:
            self.turn = Player.O
        else:
            self.turn = Player.X

    def validateEntry(self, row: int, col: int) -> bool:
        """Valid the user's entry.

        If the user's entry is valid, return True.
        Else, return False.
        """
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
        """Check the board is full.

        If the board is full, return True.
        Else, return False.
        """
        return all(
            cell != Cell.EMPTY for row in self.board.board for cell in row
        )

    def checkWin(self) -> bool:
        """Check if the player on this turn has won after their move.

        If the player on this turn has won, return True.
        Else, return False.
        """

        board = self.board.board
        board_size = self.board.BOARD_SIZE
        # Check row
        for row in board:
            if row[0] != Cell.EMPTY and all(cell == row[0] for cell in row):
                print("row")
                return True
        # Check column
        for col in zip(*board):
            if col[0] != Cell.EMPTY and all(
                cell == col[0] and cell != Cell.EMPTY for cell in col
            ):
                print("col")
                return True
        # Check main diagonal
        if board[0][0] != Cell.EMPTY and all(
            board[i][i] == board[0][0] for i in range(board_size)
        ):
            print("main diagonal")
            return True
        # Check anti-diagonal
        if board[board_size - 1][0] != Cell.EMPTY and all(
            board[i][board_size - 1 - i] == board[0][board_size - 1]
            for i in range(board_size)
        ):
            print("anti-diagonal")
            return True

    def checkEnd(self) -> bool:
        """Check if the game is end or not.

        If the game is ended, return True.
        Else, return False.
        """

        if self.checkWin():
            print(f"{self.turn.name} IS THE WINNER!!!")
            return True
        if self.checkFull():
            print("DRAW! NOBODY WINS!")
            return True
        return False

    def playGame(self):
        """Play a single game of Tic-Tac-Toe"""
        board = self.board.board
        print(f"\nNew Game: {self.turn.name} goes first.\n")

        while True:
            print(f"{self.turn.name}'s turn.")
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
            print("Thank you for your selection.")
            self.board.printBoard()
            if self.checkEnd():
                return
            self.switchPlayer()


def main():
    """Main game loop"""
    game = Game()
    while True:
        game.playGame()
        user_input = input("Another game? Enter Y or y for yes.\n")
        if user_input.lower() != "y":
            break
    print("Thanks for playing!")


if __name__ == "__main__":
    main()
