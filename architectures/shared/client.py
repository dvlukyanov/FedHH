import time
from typing import Optional

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from architectures.shared.config import Config
from architectures.shared.proxy import ProxyPool, Proxy
from architectures.shared.protocol import Command, CommandAction, CommandResponse


__author__ = 'Dmitry Lukyanov'
__email__ = 'dmitry@dmitrylukyanov.com'
__license__ = 'MIT'


class Client():

    def __init__(self, id, model_type, data):
        self.id = id
        self.model_type = model_type
        self.data = data

    def train(self, model_src, model_target):
        proxy: Optional[Proxy] = None
        while not proxy:
            proxy = ProxyPool().acquire()
            if not proxy:
                time.sleep(1)
        print(f'Proxy {proxy} is acquired by {self}')
        command = Command(
            action=CommandAction.TRAIN,
            model_type=self.model_type,
            folder=Config()['storage']['models'],
            model_src=model_src,
            model_target=model_target,
            epoch=Config()['client']['training']['epochs'],
            batch_size=Config()['client']['training']['batch_size'],
            items=self.data,
            test_ratio=0.2,
            seed=Config()['seed']
        )
        response: CommandResponse = proxy.execute(command)
        ProxyPool().release(proxy)
        return response


class ClientPool():

    def __init__(self):
        self.clients = set()

    def create(self, model_type, data):
        id = max([client.id for client in self.clients]) + 1 if len(self.clients) > 0 else 0
        client = Client(id, model_type, data)
        self.clients.add(client)
        print(f'Client {client} added to the pool')