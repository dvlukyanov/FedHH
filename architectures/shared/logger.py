import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from architectures.shared.config import Config


__author__ = 'Dmitry Lukyanov'
__email__ = 'dmitry@dmitrylukyanov.com'
__license__ = 'MIT'


class Logger():

    def __init__(self):
        self.folder = Config()['log']['folder']

    def server(self, msg):
        self.__log('server.log', msg)

    def edge(self, msg):
        self.__log('edge.log', msg)

    def client(self, msg):
        self.__log('client.log', msg)

    def proxy(self, msg):
        self.__log('proxy.log', msg)

    def worker(self, msg):
        self.__log('worker.log', msg)

    def __log(self, file, msg):
        with open(self.folder + '/' + file, 'a') as f: 
            f.write(msg + '\n')
        print(msg)