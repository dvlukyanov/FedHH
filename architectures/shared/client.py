import time
from typing import Optional

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from architectures.shared.config import Config
from architectures.shared.proxy import ProxyPool, Proxy
from architectures.shared.protocol import Command, CommandAction, CommandResponse
from architectures.shared.logger import Logger


__author__ = 'Dmitry Lukyanov'
__email__ = 'dmitry@dmitrylukyanov.com'
__license__ = 'MIT'


class Client():

    def __init__(self, id, model_type, data):
        self.id = id
        self.model_type = model_type
        self.data = data
        Logger.client(f'Client {self.id} is initialized')

    def train(self, model_src, model_target):
        proxy: Optional[Proxy] = None
        while not proxy:
            proxy = ProxyPool().acquire()
            if not proxy:
                time.sleep(1)
        Logger.client(f'Proxy {proxy.hostname} is acquired by client {self.id}')
        Logger.client(f'model_type: {self.model_type}')
        Logger.client(f'folder: {Config()["storage"]["models"]}')
        Logger.client(f'model_src: {model_src}')
        Logger.client(f'model_target: {model_target}')
        Logger.client(f'epochs: {Config()["client"]["training"]["epochs"]}')
        Logger.client(f'batch_size: {Config()["client"]["training"]["batch_size"]}')
        Logger.client(f'items: {self.data}')
        Logger.client(f'test_ratio: {0.2}')
        Logger.client(f'seed: {Config()["seed"]}')
        try:
            command = Command(
        action=CommandAction.TRAIN,
        model_type=self.model_type,
        folder=Config()['storage']['models'],
        model_src=model_src,
        model_target=model_target,
        epochs=Config()['client']['training']['epochs'],
        batch_size=Config()['client']['training']['batch_size'],
        items=self.data,
        test_ratio=0.2,
        seed=Config()['seed']
            )
        except Exception as e:
            Logger.client(f'Error while creating command: {e}')
        Logger.client(f'Command {command} is formed at client {self.id}')
        response: CommandResponse = proxy.execute(command)
        Logger.client(f'Command {command} is send to proxy at client {self.id}')
        ProxyPool().release(proxy)
        Logger.client(f'Proxy {self.hostname} is released by client {self.id}')
        return response


class ClientPool():

    def __init__(self):
        self.clients = set()

    def create(self, model_type, data):
        id = max([client.id for client in self.clients]) + 1 if len(self.clients) > 0 else 0
        client = Client(id, model_type, data)
        self.clients.add(client)
        Logger.client(f'Client {client.id} added to the pool')