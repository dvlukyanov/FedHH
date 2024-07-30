from synchronized import synchronized


__author__ = 'Dmitry Lukyanov'
__email__ = 'dmitry@dmitrylukyanov.com'
__license__ = 'MIT'


class Proxy():

    def __init__(self, id, hostname, connection):
        self.id = id
        self.hostname = hostname
        self.connection = connection
        self.available = True

    def acquire(self):
        if not self.available:
            raise RuntimeError('Proxy ' + self + ' is not available')
        self.available = False

    def release(self):
        if self.available:
            raise RuntimeError('Proxy ' + self + ' is not acquired')
        self.available = True

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
        self.proxies.append(proxy)
        print(f'Proxy {proxy} + is created and added to the pool')

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