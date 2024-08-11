import concurrent.futures
import torch

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from architectures.shared.config import Config
from architectures.shared.client import ClientPool
from models.model_factory import ModelFactory
from architectures.shared.notifier import notify_slack
from architectures.shared.utils import load_model, save_model


__author__ = 'Dmitry Lukyanov'
__email__ = 'dmitry@dmitrylukyanov.com'
__license__ = 'MIT'


class Edge():

    def __init__(self, id, model_type, data):
        self.id = id
        self.model_type = model_type
        self.data = data
        self.model_name = self._init_model()
        self.client_pool = ClientPool()
        self._setup_architecture()

    def _init_model(self):
        model = ModelFactory.create(self.model_type)
        model_name = self._create_model_name(round, 'initialized')
        save_model(model, self.model_type, Config()['storage']['models'], model_name)
        return model_name

    def _create_model_name(self, round, iteration):
        return Config()['edge']['models']['name'].format(round=round, iteration=iteration, edge_id=self.id)
    
    def _setup_architecture(self):
        data = None # TODO
        for id in range(Config()['client']['qnt']):
            self.client_pool.create(self.model_type, data[id])

    def train(self):
        for iteration in range(Config()['edge']['training']['iterations']):
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(self.client_pool.clients)) as executor:
                futures = [executor.submit(client.train) for client in self.client_pool.clients]
                for future in concurrent.futures.as_completed(futures):
                    future.result()
            model = self._aggregate()
            self._evaluate(model)
            if self.config['notification']['enabled']:
                notify_slack(self.config['notification']['slack'], f'Epoch {epoch} completed. Training accuracy: {accuracy}. Validation accuracy: {accuracy}')
            self._distribute()

    def _aggregate(self):
        names = [client.model_name for client in self.client_pool.clients]
        models = [load_model(self.model_type, Config()['storage']['models'], model_name) for model_name in names]
        states = [model.state_dict() for model in models]
        model = ModelFactory.create(self.model_type)
        state = {}
        for key in states[0].keys():
            state[key] = torch.mean(torch.stack([state[key].to(torch.float32) for state in states]), dim=0, keepdim=False)
        model.load_state_dict(state)
        model.eval()
        return model

    def _evaluate(self):
        pass

    
class EdgePool():

    def __init__(self):
        self.edges = []

    def create(self, model_type, data):
        id = max([edge.id for edge in self.edges]) + 1 if len(self.edges) > 0 else 0
        edge = Edge(id, model_type, data)
        self.edges.append(edge)
        print(f'Edge {edge} was created and added to the pool')