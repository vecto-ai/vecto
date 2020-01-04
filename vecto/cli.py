import argparse
import vecto


class CLI(object):

    def __init__(self):
        parser = argparse.ArgumentParser(
            prog="vecto",
            description='vecto commad line interface',
            add_help=True,
            epilog="\n",
            usage='''vecto <command> [<args>],

The most commonly used vecto commands are:
   benchmark        Run benchmarks
   create_vocab     Create vocabulary from a folder
''')

        parser.add_argument('--version', action='version',
                            version=f'Vecto version {vecto.__version__}')
        parser.add_argument('command', help='Subcommand to run')
        args, self.unknownargs = parser.parse_known_args()
        if not hasattr(self, args.command):
            print('Unrecognized command')
            parser.print_help()
            exit(1)
        # use dispatch pattern to invoke method with same name
        getattr(self, args.command)()

    def benchmark(self):
        from vecto.benchmarks import run_benchmarks_cli
        run_benchmarks_cli(self.unknownargs)

    def create_vocab(self):
        print("CLI for vocabulary routines not implemented yet")


def main():
    CLI()


if __name__ == '__main__':
    main()
