import os
import time
import socket
import json
from dataclasses import asdict
import random
import torch
from architectures.shared.protocol import CommandAction, Command, CommandResult, CommandResponse
from models.model_factory import ModelFactory


__author__ = 'Dmitry Lukyanov'
__email__ = 'dmitry@dmitrylukyanov.com'
__license__ = 'MIT'


class Worker():

    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.socket = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    def start(self, delay=0):
        self._delay(delay)
        self._connect()
        self._work()

    def _delay(self, delay):
        for second in range(delay, 0, -1):
            print(f'Delaying start for {second} seconds...')
            time.sleep(1)

    def _connect(self):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.connect((self.host, self.port))
        print(f"Connected to server at {self.host}:{self.port}")
    
    def _work(self):
        try:
            while True:
                command: Command = self._receive_command()
                if command is None:
                    continue
                match command.action:
                    case CommandAction.TRAIN:
                        self._train(command)
                        self._send_response(CommandResult.DONE)
                    case CommandAction.STOP:
                        break
                    case _:
                        raise RuntimeError(f'Unknown command: {command}')
        except Exception as e:
            print(f"Error in the work loop: {e}")
        finally:
            self.socket.close()
            print(f'Worker stopped')

    def _receive_command(self):
        data = self.socket.recv(1024)
        if not data:
            return None
        command: Command = Command(**json.loads(data.decode('utf-8')))
        print(f'Command is received: {command}')
        return command

    def _train(self, command: Command):
        self._set_seed(command.seed)
        model = self._load_model(command.model_type, command.folder, command.model_src)
        criterion = model.get_criterion()
        optimizer = model.get_optimizer()
        scheduler = model.get_scheduler()

        for epoch in range(command.epochs):
            train_loss = self._train_model(model, criterion, optimizer, scheduler, train_loader)
            test_loss, test_accuracy = self._test_model(model, criterion, test_loader)
            # TODO report

    def _train_model(self, model, criterion, optimizer, scheduler, data_loader):
        running_loss = 0.0
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            optimizer.zero_grad()
            outputs = model(inputs)
            outputs = model.get_logits(outputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        if scheduler:
            scheduler.step()
        loss = running_loss / len(data_loader)
        return loss

    def _test_model(self, model, criterion, data_loader):
        running_loss = 0.0
        correct = 0
        total = 0
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
        accuracy = 100 * correct / total
        return running_loss / len(data_loader), accuracy
    
    def _set_seed(self, seed):
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def _load_model(self, model_type, folder, model_name):
        model_path = os.path.join(folder, model_name)
        model = ModelFactory.create_model(model_type).to(self.device)
        model.load_state_dict(torch.load(model_path))
        model.eval()
        print(f"Model {model_name} ({model_type}) is loaded")
        return model
    
    def _save_model(self, model, model_type, folder, model_name):
        model_path = os.path.join(folder, model_name)
        torch.save(model.state_dict(), model_path)
        print(f'Model {model_path} ({model_type}) is saved')

    def _send_response(self, result: CommandResponse):
        response = CommandResponse(result=result)
        data = json.dumps(asdict(response)).encode('utf-8')
        self.socket.sendall(data)
        print(f'Response is sent to the server: {response}')