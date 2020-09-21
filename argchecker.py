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

        assert not(args.history_to_use != "all" and int(args.history_to_use) <
                   args.future_bars), "It is a good idea to use more history and less future bars. Please change these two values and try again.\nExiting now..."

        args.market_index = str(args.market_index).upper()
        if args.history_to_use != "all":
            args.history_to_use = int(args.history_to_use)
