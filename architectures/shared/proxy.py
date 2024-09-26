import json
from enum import Enum
from dataclasses import asdict
from typing import Any, Dict
import pandas as pd
import numpy as np

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from architectures.shared.synchronized import synchronized
from architectures.shared.protocol import Command, CommandResponse, CommandResult, Metric
from architectures.shared.logger import Logger


__author__ = 'Dmitry Lukyanov'
__email__ = 'dmitry@dmitrylukyanov.com'
__license__ = 'MIT'


class Proxy():

    def __init__(self, id, hostname, connection):
        self.id = id
        self.hostname = hostname
        self.connection = connection
        self.available = True
        Logger.proxy(f'Proxy {self.hostname} is initialized')

    def acquire(self):
        if not self.available:
            raise RuntimeError('Proxy ' + self + ' is not available')
        self.available = False
        Logger.proxy(f'Proxy {self.hostname} is acquired')

    def release(self):
        if self.available:
            raise RuntimeError('Proxy ' + self.hostname + ' is not acquired')
        self.available = True
        Logger.proxy(f'Proxy {self.hostname} is released')

    def execute(self, command: Command):
        data = self._serialize(command)
        Logger.proxy(data)
        try:
            self.connection.sendall(data.encode('utf-8'))
            Logger.proxy(f"Command was sent to the worker {self.hostname}: {command}")
        except Exception as e:
            Logger.proxy(f"Error sending data: {e}")
        while True:
            response: CommandResponse = self._receive_response()
            if response is None:
                continue
            match response.result:
                case CommandResult.DONE:
                    return response
                case _:
                    raise RuntimeError(f'Unknown result: {command}')
                
    def _serialize(self, command: Any) -> str:
        def convert(value):
            if isinstance(value, pd.DataFrame):
                return value.to_dict(orient='records')
            elif isinstance(value, Enum):
                return value.name
            return value
        command_dict = asdict(command)
        serializable_dict = {k: convert(v) for k, v in command_dict.items()}
        return json.dumps(serializable_dict)
                
    def _receive_response(self):
        try:
            data = self.connection.recv(1024 * 1024 * 1024)
        except Exception as e:
            Logger.proxy(f"Error receiving data: {e}")
        if not data:
            return None
        response: CommandResponse = self._deserialize(data)
        print(f'Response is received: {response}')
        return response
    
    def _deserialize(self, data: str) -> Command:
        def _deserialize_metric(self, metric_dict: dict) -> Metric:
            if 'cmatrix' in metric_dict and isinstance(metric_dict['cmatrix'], list):
                metric_dict['cmatrix'] = np.array(metric_dict['cmatrix'])
            return Metric(**metric_dict)
        data_dict = json.loads(data.decode('utf-8'))
        if 'result' in data_dict:
            data_dict['result'] = CommandResponse[data_dict['result']]
        if 'items' in data_dict:
            data_dict['items'] = pd.DataFrame(data_dict['items'])
        if 'train_history' in data_dict:
            data_dict['train_history'] = [self._deserialize_metric(m) for m in data_dict['train_history']]
        if 'test_history' in data_dict:
            data_dict['test_history'] = [self._deserialize_metric(m) for m in data_dict['test_history']]
        return Command(**data_dict)

    def __members(self):
        return (self.id)

    def __eq__(self, other):
        if type(other) is type(self):
            return self.__members() == other.__members()
        else:
            return False

    def __hash__(self):
        return hash(self.__members())


class ProxyPool():

    _instance = None

    def __new__(cls, limit=None):
        if not cls._instance:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, limit=None):
        if not hasattr(self, 'initialized'):
            self.proxies = set()
            self.limit = limit
            self.initialized = True

    @synchronized
    def create(self, hostname, connection):
        if len(self.proxies) + 1 > self.limit:
            raise RuntimeError('The proxy pool is full: ' + len(self.proxies))
        id = max([proxy.id for proxy in self.proxies]) + 1 if len(self.proxies) > 0 else 0
        proxy = Proxy(id=id, hostname=hostname, connection=connection)
        self.proxies.add(proxy)
        Logger.proxy(f'Proxy {proxy.hostname} is created and added to the pool')

    @synchronized
    def acquire(self):
        proxy = next((proxy for proxy in self.proxies if proxy.available), None)
        if not proxy:
            return None
        proxy.acquire()
        return proxy
    
    @synchronized
    def release(self, proxy):
        if proxy not in self.proxies:
            raise RuntimeError('There is no such proxy in the pool: ' + proxy)
        proxy.release()
