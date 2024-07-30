import yaml


class Config:
    
    _instance = None

    def __new__(cls, path=None):
        if not cls._instance:
            cls._instance = super().__new__(cls)
            cls._instance._initialize(path)
        return cls._instance

    def _initialize(self, path):
        if path:
            self.config_data = self._load_config(path)
        else:
            self.config_data = {}

    def _load_config(self, path):
        with open(path, 'r') as file:
            config = yaml.safe_load(file)
            print('Configuration is loaded')
            return config
        
    def __getitem__(self, key):
        return self.config_data[key]

    def __setitem__(self, key, value):
        self.config_data[key] = value
