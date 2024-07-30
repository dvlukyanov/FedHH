import argparse
import socket
import time
import concurrent.futures
from architectures.shared.config import Config
from architectures.shared.proxy import ProxyPool
from architectures.shared.notifier import notify_slack


__author__ = 'Dmitry Lukyanov'
__email__ = 'dmitry@dmitrylukyanov.com'
__license__ = 'MIT'


class Client():

    def __init__(self, id, data_subset):
        self.id = id
        self.data_subset = data_subset

    def train(self, model_name):
        proxy = None
        while not proxy:
            proxy = ProxyPool().acquire()
            if not proxy:
                time.sleep(1)
        print(f'Proxy {proxy} is acquired by {self}')
        # send to the proxy - a model name, a data subset
        # wait until the work is done 
        ProxyPool().release(proxy)
        # return result


class ClientPool():

    def __init__(self):
        self.clients = set()

    def add_client(self):
        id = max([client.id for client in self.clients]) + 1 if len(self.clients) > 0 else 0
        client = Client(id)
        self.clients.add(client)
        print(f'Client {client} added to the pool')


class Edge():

    def __init__(self, id, model_type):
        self.id = id
        self.model_type = model_type
        self._init_model()
        self.client_pool = ClientPool()

    def _init_model(self):
        # for each edge initiate a model
        pass

    def train(self):
        for iteration in range(Config()['edge']['training']['iterations']):
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(self.client_pool.clients)) as executor:
                futures = [executor.submit(client.train) for client in self.client_pool.clients]
                for future in concurrent.futures.as_completed(futures):
                    future.result()
            self._aggregate()
            self._evaluate()
            if self.config['notification']['enabled']:
                notify_slack(self.config['notification']['slack'], f'Epoch {epoch} completed. Training accuracy: {accuracy}. Validation accuracy: {accuracy}')
            self._distribute()

    def _aggregate(self):
        # TODO
        pass

    
class EdgePool():

    def __init__(self):
        self.edges = []

    def create(self):
        id = max([edge.id for edge in self.edges]) + 1 if len(self.edges) > 0 else 0
        edge = Edge(id)
        self.edges.append(edge)
        print(f'Edge {edge} was created and added to the pool')


class Server():

    def __init__(self, config, host, port, worker_qnt):
        self.config = config
        self.host = host
        self.port = port
        self.edge_pool = EdgePool()
        ProxyPool(worker_qnt)

    def serve(self):
        server_socket = self._listen()
        self._setup_workers(server_socket)
        self._setup_architecture()
        self._train()
        self._notify()

    def _listen(self):
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.bind((self.host, self.port))
        server_socket.listen()
        print(f"Server is listening on {self.host}:{self.port}")
        return server_socket

    def _setup_workers(self, server_socket):
        while len(ProxyPool().workers) < ProxyPool().limit:
            conn, addr = server_socket.accept()
            ProxyPool().create(hostname=addr, connection=conn)
        print(f'All workers are connected: {len(ProxyPool().workers)}')


    def _setup_architecture(self):
        for id in range(Config()['edge']['qnt']):
            model_type = Config()['edge']['models'][id % len(Config()['edge']['models'])]
            edge = Edge(id, model_type)
            # for each edge create M clients
            # for each client allocate a subset of the data

    def _train(self):
        for round in range(Config()['server']['rounds']):
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(self.edge_pool.edges)) as executor:
                futures = [executor.submit(edge.train) for edge in self.edge_pool.edges]
                for future in concurrent.futures.as_completed(futures):
                    future.result()
            self._extract()
            self._evaluate()
            if self.config['notification']['enabled']:
                notify_slack(self.config['notification']['slack'], f'Epoch {epoch} completed. Training accuracy: {accuracy}. Validation accuracy: {accuracy}')
            # TODO distribute

    def _extract(self):
        # TODO
        pass

    def _evaluate(self):
        # TODO
        pass

    def _notify(self):
        # TODO
        pass


def start_server(host='0.0.0.0', port=12345, workers=0):
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((host, port))
    server_socket.listen()
    print(f"Server listening on {host}:{port}")

    cnt = 0
    
    while cnt < workers:
        client_socket, client_address = server_socket.accept()
        print(f"Connection from {client_address} has been established!")
        
        data = client_socket.recv(1024)
        print(f"Received data: {data.decode('utf-8')}")
        
        response = "Hello, client!"
        client_socket.sendall(response.encode('utf-8'))
        
        client_socket.close()
        cnt += 1


def start_worker(host='127.0.0.1', port=12345, hostname='127.0.0.1'):
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((host, port))
    print(f"Connected to server at {host}:{port}")
    client_socket.sendall(("Hello from " + hostname + ", server!").encode('utf-8'))
    
    response = client_socket.recv(1024)
    print(f"Received response: {response.decode('utf-8')}")
    
    client_socket.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--server', type=str, default='')
    parser.add_argument('--port', type=str, default='')
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--slack', type=str, default=None)
    args = parser.parse_args()

    config = Config('fedhh.yaml')
    if config['notification']['enabled'] and args.slack:
        config['notification']['slack'] = args.slack

    hostname = socket.gethostname()

    print(f'Server: {args.server}')
    print(f'Hostname: {hostname}')
    if args.server == hostname:
        print('Launching a server')
        Server(config, args.server, int(args.port), args.workers).serve()
    else:
        print('Launching a worker')
        time.sleep(5)
        start_worker(config, args.server, int(args.port), hostname)


if __name__ == '__main__':
    main()