import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from architectures.shared.config import Config
from architectures.shared.synchronized import synchronized


__author__ = 'Dmitry Lukyanov'
__email__ = 'dmitry@dmitrylukyanov.com'
__license__ = 'MIT'


class Logger():

    def __init__(self):
        self.folder = Config()['log']['folder']

    @synchronized
    def server(self, msg):
        self.__log('server.log', msg)

    @synchronized
    def edge(self, msg):
        self.__log('edge.log', msg)

    @synchronized
    def client(self, msg):
        self.__log('client.log', msg)

    @synchronized
    def proxy(self, msg):
        self.__log('proxy.log', msg)

    @synchronized
    def worker(self, msg):
        self.__log('worker.log', msg)

    @synchronized
    def __log(self, file, msg):
        with open(self.folder + '/' + file, 'a') as f: f.write(msg + '\n')
        print(msg)