import json
from enum import Enum
from dataclasses import asdict
from typing import Any, Dict
import pandas as pd

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from architectures.shared.synchronized import synchronized
from architectures.shared.protocol import Command, CommandResponse, CommandResult
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
        # Logger.proxy(f'Command will be sent to the worker: {command}')
        data = self._serialize(command)
        # Logger.proxy(data)
        try:
            Logger.proxy(str(type(data)))
            Logger.proxy(str(type(data.encode('utf-8'))))
            self.connection.sendall(data.encode('utf-8'))
            Logger.proxy("Command sent successfully")
        except Exception as e:
            Logger.proxy(f"Error sending data: {e}")
        Logger.proxy(f'Command is sent to the worker: {command}')
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
        data = self.connection.recv(1024)
        if not data:
            return None
        response: CommandResponse = CommandResponse(**json.loads(data.decode('utf-8')))
        print(f'Response is received: {response}')
        return response

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
