from Visualizer import Visualizer


def printBoard(board):
    """
    Prints out the board for debugging

    :param board: Board representation passed from the game
    """
    output = ""
    for row in board:
        for cell in row:
            if cell['revealed']:
                if cell['status'] == 1:
                    output += "EMPTY   | "
                elif cell['status'] == 2:
                    output += "SAFE    | "
                elif cell['status'] == 3:
                    output += "BRICK   | "
                elif cell['status'] == 4:
                    output += "HEAL    | "
                elif cell['status'] == 5:
                    output += "CHEST   | "
                elif cell['status'] == 6:
                    output += "DRAGON  | "
                elif cell['status'] == 7:
                    output += "ENEMY   | "
            else:
                output += "UNKNOWN | "

        output += '\n'

    print(output)


if __name__ == "__main__":
    """
    Currently just used to test out the Visualizer() function with manual commands
    """
    game = Visualizer()

    while True:
        choice = input("Level up (1) or click (2) > ")
        res = None
        if choice == '1':
            res = game.take_action(130)
        elif choice == '2':
            row = int(input("Enter row > "))
            col = int(input("Enter col > "))

            action = row * 13 + col
            res = game.take_action(action)

        printBoard(res['boardState'])
