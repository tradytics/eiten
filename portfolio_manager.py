# Basic libraries
import argparse
import json
from eiten import Eiten
from argchecker import ArgChecker

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

"""
Sample run:
python portfolio_manager.py --is_test 1 --future_bars 90 --data_granularity_minutes 3600 --history_to_use all --apply_noise_filtering 1 --market_index QQQ --only_long 1 --eigen_portfolio_number 3 --save_plot False
"""

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
