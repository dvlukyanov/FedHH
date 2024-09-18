import socket
import concurrent.futures
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from architectures.shared.edge import EdgePool
from architectures.shared.proxy import ProxyPool
from data.dataset import CustomImageDataset
from architectures.shared.config import Config
from architectures.shared.notifier import notify_slack
from architectures.shared.utils import split_data


__author__ = 'Dmitry Lukyanov'
__email__ = 'dmitry@dmitrylukyanov.com'
__license__ = 'MIT'


class Server():

    def __init__(self, config, host, port, worker_qnt):
        self.config = config
        self.host = host
        self.port = port
        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        self.items = None
        self.edge_pool = EdgePool()
        ProxyPool(worker_qnt)

    def serve(self):
        server_socket = self._listen()
        self._setup_workers(server_socket)
        self._setup_architecture()
        self._train()
        # self._notify()

    def _listen(self):
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.bind((self.host, self.port))
        server_socket.listen()
        print(f"Server is listening on {self.host}:{self.port}")
        return server_socket

    def _setup_workers(self, server_socket):
        while len(ProxyPool().proxies) < ProxyPool().limit:
            conn, addr = server_socket.accept()
            ProxyPool().create(hostname=addr, connection=conn)
        print(f'All workers are connected: {len(ProxyPool().proxies)}')

    def _setup_architecture(self):
        data = split_data(pd.read_csv(Config()['storage']['data']['labels']), int(Config()['edge']['qnt']))
        for id in range(int(Config()['edge']['qnt'])):
            model_type = Config()['edge']['models']['list'][id % len(Config()['edge']['models']['list'])]
            self.edge_pool.create(model_type, data[id])

    def _train(self):
        for round in range(Config()['server']['rounds']):
            results = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(self.edge_pool.edges)) as executor:
                futures = [executor.submit(edge.train) for edge in self.edge_pool.edges]
                for future in concurrent.futures.as_completed(futures):
                    results.append(future.result())
                for result in results:
                    print(result)
            # self._extract()
            # self._evaluate()
            # if self.config['notification']['enabled']:
                # notify_slack(self.config['notification']['slack'], f'Epoch {epoch} completed. Training accuracy: {accuracy}. Validation accuracy: {accuracy}')
            # self._distribute()

    def _extract(self):
        optimizer = model.optimizer(model)
        scheduler = model.scheduler(optimizer)
        train_loader, test_loader = self._get_data_loaders()

        softmax_temperature = Config()['server']['knowledge_transfer']['temperature']
        criterion_kldl = torch.nn.KLDivLoss(reduction="batchmean")

        for epoch in range(Config()['server']['knowledge_transfer']['epochs']):
            running_loss = 0.0

            correct_predictions_train = 0
            total_samples_train = 0
            all_labels = []
            all_predictions = []

            for i, data in enumerate(train_loader, 0):
                inputs, labels = data[0].to(self.device), data[1].to(self.device)
                
                optimizer.zero_grad()

                teacher_outputs = self.get_averaged_logits(list(map(lambda c: c.model, clients)), inputs)
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
                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predicted_train.cpu().numpy())

                running_loss += loss.item()

            accuracy_train = 100 * correct_predictions_train / total_samples_train
            precision = precision_score(all_labels, all_predictions, average='weighted')
            recall = recall_score(all_labels, all_predictions, average='weighted')
            f1 = f1_score(all_labels, all_predictions, average='weighted')
            conf_matrix = confusion_matrix(all_labels, all_predictions)

            correct_predictions_test = 0
            total_samples_test = 0
            with torch.no_grad():
                for data in test_loader:
                    inputs, labels = data[0].to(self.device), data[1].to(self.device)
                    outputs = model(inputs)
                    _, predicted_test = torch.max(outputs, 1)
                    total_samples_test += labels.size(0)
                    correct_predictions_test += (predicted_test == labels).sum().item()

            accuracy_test = 100 * correct_predictions_test / total_samples_test

            print(f'KT: {model_type}. Epoch: {epoch + 1}. Loss: {running_loss / 100:.3f}. Train accuracy: {accuracy_train:.2f}%. Test accuracy: {accuracy_test:.2f}%')

            scheduler.step()

    def _get_data_loaders(self):
        dataset = CustomImageDataset(img_dir=Config()['storage']['data']['folder'], labels_file=Config()['storage']['data']['labels'])
        train_indices, test_indices = train_test_split(self.items, test_size=Config()['server']['knowledge_transfer']['test_ratio'])
        train_dataset = Subset(dataset, train_indices)
        test_dataset = Subset(dataset, test_indices)
        train_loader = DataLoader(train_dataset, batch_size=Config()['server']['knowledge_transfer']['batch_size'], shuffle=True, num_workers=Config()['worker']['cpu_workers'])
        test_loader = DataLoader(test_dataset, batch_size=Config()['server']['knowledge_transfer']['batch_size'], shuffle=False, num_workers=Config()['worker']['cpu_workers'])
        return train_loader, test_loader

    def _evaluate(self):
        # TODO
        pass

    def _notify(self):
        # TODO
        pass

    def _distribute(self):
        pass