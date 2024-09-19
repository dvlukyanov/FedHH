import os
import time
import socket
import json
from dataclasses import asdict
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import torch
from torch.utils.data import DataLoader, Subset

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from architectures.shared.config import Config
from architectures.shared.protocol import CommandAction, Command, CommandResult, Metric, CommandResponse
from models.model_factory import ModelFactory
from data.dataset import CustomImageDataset
from architectures.shared.logger import Logger
from architectures.shared.utils import load_model, save_model
from architectures.shared.notifier import notify_slack


__author__ = 'Dmitry Lukyanov'
__email__ = 'dmitry@dmitrylukyanov.com'
__license__ = 'MIT'


class Worker():

    def __init__(self, host, port):
        self.address = socket.gethostname()
        self.host = host
        self.port = port
        self.socket = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        Logger.worker(f'Worker {self.address} is initialized')

    def start(self):
        self._delay(int(Config()['worker']['delay']))
        self._connect()
        self._work()

    def _delay(self, delay):
        for second in range(delay, 0, -1):
            Logger.worker(f'Delaying start for {second} seconds...')
            time.sleep(1)

    def _connect(self):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.connect((self.host, self.port))
        Logger.worker(f"Connected to server at {self.host}:{self.port}")
    
    def _work(self):
        try:
            while True:
                command: Command = self._receive_command()
                if command is None:
                    continue
                match command.action:
                    case CommandAction.TRAIN:
                        model, train_history, test_history = self._train(command)
                        save_model(model, command.model_type, command.folder, command.model_target)
                        self._send_response(CommandResult.DONE, train_history, test_history)
                    case CommandAction.STOP:
                        break
                    case _:
                        raise RuntimeError(f'Unknown command: {command}')
        except Exception as e:
            Logger.worker(f"Error in the work loop: {e}")
        finally:
            self.socket.close()
            Logger.worker(f'Worker stopped')

    def _receive_command(self):
        try:
            data = self.socket.recv(1024 * 1024)
        except Exception as e:
            Logger.worker(f"Error receiving data: {e}")
        if not data:
            return None
        command: Command = self._deserialize(data)
        Logger.worker(f'Command is received: {command}')
        return command
    
    def _deserialize(self, data: str) -> Command:
        data_dict = json.loads(data.decode('utf-8'))
        if 'action' in data_dict:
            data_dict['action'] = CommandAction[data_dict['action']]
        if 'items' in data_dict:
            data_dict['items'] = data_dict['items']
        return Command(**data_dict)

    def _train(self, command: Command):
        self._set_seed(command.seed)
        model = load_model(command.model_type, command.folder, command.model_src)
        criterion = model.get_criterion()
        optimizer = model.get_optimizer()
        scheduler = model.get_scheduler()

        train_loader, test_loader = self._get_data_loaders(command)

        train_history = []
        test_history = []
        for epoch in range(command.epochs):
            train_metric: Metric = self._train_model(model, criterion, optimizer, scheduler, train_loader)
            test_metric: Metric = self._test_model(model, criterion, test_loader)
            train_history.append(train_metric)
            test_history.append(test_metric)
            notify_slack(f'Worker {self.address} trained {command.model_src} through {epoch+1} epochs. Test accuracy: {test_metric.accuracy}')
        return model, train_history, test_history
    
    def _get_data_loaders(self, command):
        dataset = CustomImageDataset(img_dir=Config()['storage']['data']['folder'], labels_file=Config()['storage']['data']['labels'])
        train_indices, test_indices = train_test_split(command.items, test_size=command.test_ratio)
        train_dataset = Subset(dataset, train_indices)
        test_dataset = Subset(dataset, test_indices)
        train_loader = DataLoader(train_dataset, batch_size=command.batch_size, shuffle=True, num_workers=Config()['worker']['cpu_workers'])
        test_loader = DataLoader(test_dataset, batch_size=command.batch_size, shuffle=False, num_workers=Config()['worker']['cpu_workers'])
        return train_loader, test_loader

    def _train_model(self, model, criterion, optimizer, scheduler, data_loader):
        running_loss = 0.0
        correct = 0
        total = 0
        all_labels = []
        all_predictions = []
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            optimizer.zero_grad()
            outputs = model(inputs)
            outputs = model.get_logits(outputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
        if scheduler:
            scheduler.step()
        loss = running_loss / len(data_loader)
        accuracy = 100 * correct / total
        precision = precision_score(all_labels, all_predictions, average='weighted')
        recall = recall_score(all_labels, all_predictions, average='weighted')
        f1 = f1_score(all_labels, all_predictions, average='weighted')
        conf_matrix = confusion_matrix(all_labels, all_predictions)
        return Metric(
            loss=loss,
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1=f1,
            cmatrix=conf_matrix
        )

    def _test_model(self, model, criterion, data_loader):
        running_loss = 0.0
        correct = 0
        total = 0
        all_labels = []
        all_predictions = []
        with torch.no_grad():
            for inputs, labels in data_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = model(inputs)
                outputs = model.get_logits(outputs)
                loss = criterion(outputs, labels)
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())
        loss = running_loss / len(data_loader)
        accuracy = 100 * correct / total
        precision = precision_score(all_labels, all_predictions, average='weighted')
        recall = recall_score(all_labels, all_predictions, average='weighted')
        f1 = f1_score(all_labels, all_predictions, average='weighted')
        conf_matrix = confusion_matrix(all_labels, all_predictions)
        return Metric(
            loss=loss,
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1=f1,
            cmatrix=conf_matrix
        )
    
    def _set_seed(self, seed):
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def _send_response(self, result: CommandResponse, train_history, test_history):
        response = CommandResponse(result=result, train_history=train_history, test_history=test_history)
        data = json.dumps(asdict(response)).encode('utf-8')
        self.socket.sendall(data)
        Logger.worker(f'Response is sent to the server: {response}')