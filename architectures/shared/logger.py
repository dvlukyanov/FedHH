import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from architectures.shared.config import Config


__author__ = 'Dmitry Lukyanov'
__email__ = 'dmitry@dmitrylukyanov.com'
__license__ = 'MIT'


class Logger():

    @classmethod
    def server(self, msg):
        self.__log('server.log', msg)

    @classmethod
    def edge(self, msg):
        self.__log('edge.log', msg)

    @classmethod
    def client(self, msg):
        self.__log('client.log', msg)

    @classmethod
    def proxy(self, msg):
        self.__log('proxy.log', msg)

    @classmethod
    def worker(self, msg):
        self.__log('worker.log', msg)

    @classmethod
    def __log(self, file, msg):
        with open(Config()['log']['folder'] + '/' + file, 'a') as f: 
            f.write(msg + '\n')
        print(msg)