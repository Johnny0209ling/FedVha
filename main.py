from args import args_parser
from server import FedAvg


def main():
    args = args_parser()
    trainer = FedAvg(args)
    trainer.server()


if __name__ == "__main__":
    main()
