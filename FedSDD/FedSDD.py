import sys
sys.path.insert(0, '../')
import os
import random
import math
from statistics import mean
import yaml
import pickle as pkl
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from model.model_factory import ModelFactory
from util.dataload import ImageDataset


__author__ = 'Huaye Li'
__email__ = 'huayel@g.clemson.edu'
__license__ = 'MIT'


class Server:
    def __init__(self, config):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        self.config = config
        self.rng = random.Random()
        self.rng.seed(self.config['seed'])
        self.round = 0
        self.clients = [Client(self, idx, self.device) for idx in range(1, self.config['client']['qnt'] + 1)]
        self.model_types = list(set([client.model_type for client in self.clients]))
        self.models = {type: {'groups': [{'name': None, 'model': None} for _ in range(self.config['server']['group_num'])]} for type in self.model_types}
        self.accuracy = {}

    def train(self):
        rounds = math.ceil(self.config['server']['epochs'] / (self.config['client']['training']['epochs']))
        for round in range(1, rounds + 1):
            clients = self.select_clients()
            print(f'Chosen clients: {[client.id for client in clients]}')
            for client in clients:
                client.train(self.models[client.model_type]['groups'][client.group]['name'])
            
            for type in self.models.keys():
                for group in range(self.config['server']['group_num']):
                    self.models[type]['groups'][group] = self.aggregate(clients, type, group)
                
                self.models[type]['groups'][0] = self.extract(clients, type)  # Only extract for main group
            
            accuracy = self.test()
            self.save_models()
            self.round = round
            epochs = self.round * self.config["client"]["training"]["epochs"]
            for type, acc in accuracy.items():
                if type not in self.accuracy:
                    self.accuracy[type] = {}
                self.accuracy[type][epochs] = acc
            
    def select_clients(self):
        groups = {type: [[] for _ in range(self.config['server']['group_num'])] for type in self.model_types}
        for client in self.clients:
            groups[client.model_type][client.group].append(client)

        for type in groups:
            for group in groups[type]:
                self.rng.shuffle(group)

        selected = []
        for type in groups:
            for group_idx, group in enumerate(groups[type]):
                if group:
                    selected.append(self.rng.choice(group))
                    selected[-1].group = group_idx
                if len(selected) == self.config['server']['knowledge_transfer']['client_qnt']:
                    break
            if len(selected) == self.config['server']['knowledge_transfer']['client_qnt']:
                break
        return selected

    def aggregate(self, clients, model_type, group):
        states = [client.model.state_dict() for client in clients if client.model_type == model_type and client.group == group]
        if len(states) == 0:
            return self.models[model_type]['groups'][group]
        model = ModelFactory.create(model_type, self.device)
        state = {}
        for key in states[0].keys():
            state[key] = torch.mean(torch.stack([state[key].to(torch.float32) for state in states]), dim=0, keepdim=False)
        model.load_state_dict(state)
        model.eval()
        name = self.config['storage']['model']['name']['server'].format(model_type, group, self.round)
        print(f'Models for {model_type} group {group} are aggregated')
        return {'name': name, 'model': model}

    def extract(self, clients, model_type):
        random.seed(self.config['seed'])
        torch.manual_seed(self.config['seed'])
        torch.cuda.manual_seed(self.config['seed'])
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        model = self.models[model_type]['groups'][0]['model']
        optimizer = model.optimizer(model)
        scheduler = model.scheduler(optimizer)

        data_file = open(os.path.join(self.config['storage']['data']['path'], 'kt.pkl'), 'rb')
        data = pkl.load(data_file)
        
        train_dataset = ImageDataset(data['train'], self.config['seed'])
        test_dataset = ImageDataset(data['test'], self.config['seed'])

        trainloader = DataLoader(train_dataset, batch_size=self.config['server']['knowledge_transfer']['batch_size'], shuffle=False)
        testloader = DataLoader(test_dataset, batch_size=self.config['server']['knowledge_transfer']['batch_size'], shuffle=False)

        softmax_temperature = self.config['server']['knowledge_transfer']['temperature']
        criterion_kldl = torch.nn.KLDivLoss(reduction="batchmean")

        for epoch in range(self.config['server']['knowledge_transfer']['epochs']):
            running_loss = 0.0

            correct_predictions_train = 0
            total_samples_train = 0

            for i, data in enumerate(trainloader, 0):
                inputs, labels = data[0].to(self.device), data[1].to(self.device)
                
                optimizer.zero_grad()

                teacher_outputs = self.get_ensemble_logits(model_type, inputs)
                outputs = model(inputs)

                loss = (softmax_temperature ** 2) * criterion_kldl(
                    torch.nn.functional.log_softmax(
                        outputs / softmax_temperature, dim=1
                    ),
                    torch.nn.functional.softmax(
                        teacher_outputs / softmax_temperature,
                        dim=1,
                    ),
                )
                loss.backward()
                optimizer.step()

                _, predicted_train = torch.max(outputs, 1)
                total_samples_train += labels.size(0)
                correct_predictions_train += (predicted_train == labels).sum().item()

                running_loss += loss.item()

            accuracy_train = 100 * correct_predictions_train / total_samples_train

            correct_predictions_test = 0
            total_samples_test = 0
            with torch.no_grad():
                for data in testloader:
                    inputs, labels = data[0].to(self.device), data[1].to(self.device)
                    outputs = model(inputs)
                    _, predicted_test = torch.max(outputs, 1)
                    total_samples_test += labels.size(0)
                    correct_predictions_test += (predicted_test == labels).sum().item()

            accuracy_test = 100 * correct_predictions_test / total_samples_test

            print(f'KT: {model_type}. Epoch: {epoch + 1}. Loss: {running_loss / 100:.3f}. Train accuracy: {accuracy_train:.2f}%. Test accuracy: {accuracy_test:.2f}%')

            scheduler.step()

        return {'name': self.config['storage']['model']['name']['server'].format(model_type, 0, self.round + 1), 'model': model}

    def get_ensemble_logits(self, model_type, inputs):
        outputs = []
        for r in range(self.config['server']['temporal_ensemble_rounds']):
            for group in range(self.config['server']['group_num']):
                if self.round - r >= 0:
                    model = self.models[model_type]['groups'][group]['model']
                    with torch.no_grad():
                        _output = model(inputs)
                        outputs.append(_output)
        avg = torch.mean(torch.stack(outputs, dim=0), dim=0, keepdim=False)
        return avg

    def test(self):
        result = {}
        for type in self.models:
            data_file = open(os.path.join(self.config['storage']['data']['path'], 'test.pkl'), 'rb')
            data = pkl.load(data_file)
            test_dataset = ImageDataset(data, self.config['seed'])
            testloader = DataLoader(test_dataset, batch_size=64, shuffle=False)

            correct = 0
            total = 0
            with torch.no_grad():
                for i, data in enumerate(testloader, 0):
                    images, labels = data[0].to(self.device), data[1].to(self.device)
                    outputs = self.models[type]['groups'][0]['model'](images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            print(f'Accuracy of {type} on the {total} test images at {self.round + 1} round: {100 * correct / total} %')
            accuracy = 100 * correct / total
            result[type] = accuracy
        return result

    def save_models(self):
        for type in self.models:
            for group in range(self.config['server']['group_num']):
                item = self.models[type]['groups'][group]
                model_name = item['name']
                model_path = os.path.join(self.config['storage']['model']['path'], model_name)
                Utils.save_model(item['model'], model_path)

class Client:
    def __init__(self, server, id, device):
        self.server = server
        self.id = id
        self.device = device
        self.model_type = self.server.config['client']['models'][self.id % len(self.server.config['client']['models'])]
        self.group = (self.id - 1) % self.server.config['server']['group_num']
        self.model = ModelFactory.create(self.model_type, self.device, self.server.config['seed'])
        self.model_name = self.save_model()

    def save_model(self):
        model_name = self.server.config['storage']['model']['name']['client'].format(self.server.round, self.id)
        model_path = os.path.join(self.server.config['storage']['model']['path'], model_name)
        Utils.save_model(self.model, model_path)
        return model_name

    def train(self, model_name=None):
        random.seed(self.server.config['seed'])
        torch.manual_seed(self.server.config['seed'])
        torch.cuda.manual_seed(self.server.config['seed'])
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        if model_name:
            self.model = Utils.load_model(self.device, self.model_type, self.server.config['storage']['model']['path'] + '/' + model_name, self.server.config['seed'])

        criterion = nn.CrossEntropyLoss()
        optimizer = self.model.optimizer(self.model)
        scheduler = self.model.scheduler(optimizer)

        data_file = open(os.path.join(self.server.config['storage']['data']['path'], f'partition_{self.id}.pkl'), 'rb')
        data = pkl.load(data_file)

        train_dataset = ImageDataset(data['train'], self.server.config['seed'])
        test_dataset = ImageDataset(data['test'], self.server.config['seed'])

        trainloader = DataLoader(train_dataset, batch_size=self.server.config['client']['training']['batch_size'], shuffle=False)
        testloader = DataLoader(test_dataset, batch_size=self.server.config['client']['training']['batch_size'], shuffle=False)

        for epoch in range(self.server.config['client']['training']['epochs']):
            running_loss = 0.0
            correct_predictions_train = 0
            total_samples_train = 0

            for _, data in enumerate(trainloader, 0):
                inputs, labels = data[0].to(self.device), data[1].to(self.device)

                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                _, predicted_train = torch.max(outputs, 1)
                total_samples_train += labels.size(0)
                correct_predictions_train += (predicted_train == labels).sum().item()

                running_loss += loss.item()

            accuracy_train = 100 * correct_predictions_train / total_samples_train

            correct_predictions_test = 0
            total_samples_test = 0
            with torch.no_grad():
                for data in testloader:
                    inputs, labels = data[0].to(self.device), data[1].to(self.device)
                    outputs = self.model(inputs)
                    _, predicted_test = torch.max(outputs, 1)
                    total_samples_test += labels.size(0)
                    correct_predictions_test += (predicted_test == labels).sum().item()

            accuracy_test = 100 * correct_predictions_test / total_samples_test

            print(f'Round: {self.server.round + 1}. Client: {self.id} ({self.model_type}). Group: {self.group}. Epoch: {epoch + 1}. Loss: {running_loss / 100:.3f}. Train accuracy: {accuracy_train:.2f}%. Test accuracy: {accuracy_test:.2f}%')

            scheduler.step()


class Utils:

    @staticmethod
    def load_model(device, model_type, model_path, seed):
        model = ModelFactory.create(model_type, device, seed)
        model.load_state_dict(torch.load(model_path))
        model.eval()
        return model
    
    @staticmethod
    def save_model(model, model_path):
        torch.save(model.state_dict(), model_path)
        print(f'Model {model_path} saved')

    @staticmethod
    def test(device, model, batch_size, seed):
        data_file = open(os.path.join(config['storage']['data']['path'], 'test.pkl'), 'rb')
        data = pkl.load(data_file)
        test_dataset = ImageDataset(data, seed)
        testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        correct = 0
        total = 0
        with torch.no_grad():
            for _, data in enumerate(testloader, 0):
                images, labels = data[0].to(device), data[1].to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        return [correct, total]
    

if __name__ == "__main__":
    with open('fesdd_config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    cloud = Server(config)
    cloud.train()