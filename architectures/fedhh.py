import argparse
import socket

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from architectures.shared.server import Server
from architectures.shared.config import Config
from architectures.shared.worker import Worker


__author__ = 'Dmitry Lukyanov'
__email__ = 'dmitry@dmitrylukyanov.com'
__license__ = 'MIT'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--server', type=str, default='')
    parser.add_argument('--port', type=str, default='')
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--slack', type=str, default=None)
    parser.add_argument('--data', type=str, default=None)
    args = parser.parse_args()

    config = Config(os.path.abspath(os.path.join(os.path.dirname(__file__), 'fedhh.yaml')))
    if args.data:
        config['storage']['data']['folder'] = args.data + '/images'
        config['storage']['data']['labels'] = args.data + '/labels.csv'
    if config['notification']['enable'] and args.slack:
        config['notification']['slack'] = args.slack

    hostname = socket.gethostname()

    print(f'Server: {args.server}')
    print(f'Hostname: {hostname}')
    if args.server == hostname:
        print('Launching a server')
        Server(config, args.server, int(args.port), args.workers).serve()
    else:
        print('Launching a worker')
        Worker(host=args.server, port=int(args.port)).start()


if __name__ == '__main__':
    main()