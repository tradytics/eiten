# Basic libraries
import argparse
import json
from eiten import Eiten

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


# Argument parsing


"""
Sample run:
python portfolio_manager.py --is_test 1 --future_bars 90 --data_granularity_minutes 3600 --history_to_use all --apply_noise_filtering 1 --market_index QQQ --only_long 1 --eigen_portfolio_number 3
"""


class ArgChecker:
    """
    Argument checker
    """

    def __init__(self, args):
        print("Checking arguments...")
        self.check_arguments(args)

    def check_arguments(self, args):
        granularity_constraints_list = [1, 5, 10, 15, 30, 60, 3600]
        granularity_constraints_list_string = ''.join(
            str(value) + "," for value in granularity_constraints_list).strip(",")

        assert not(args.data_granularity_minutes not in granularity_constraints_list), "You can only choose the following values for 'data_granularity_minutes' argument -> %s\nExiting now..." % granularity_constraints_list_string

        assert not(args.is_test == 1 and args.future_bars <
                   2), "You want to test but the future bars are less than 2. That does not give us enough data to test the model properly. Please use a value larger than 2.\nExiting now..."

        assert not(args.history_to_use != "all" and int(args.history_to_use_int) <
                   args.future_bars), "It is a good idea to use more history and less future bars. Please change these two values and try again.\nExiting now..."


def main():

    argParser = argparse.ArgumentParser()
    commands = json.load(open("commands.json", "r"))
    for i in commands:
        arg_types = {"str": str, "int": int, "bool": bool}
        argParser.add_argument(i["comm"], type=arg_types[i["type"]],
                               default=i["default"], help=i["help"])

    # Get arguments
    args = argParser.parse_args()

    # Check arguments
    ArgChecker(args)

    # Run strategies
    eiten = Eiten(args)
    eiten.run_strategies()


if __name__ == '__main__':
    main()
