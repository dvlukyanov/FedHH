import argparse
import socket
import random


def start_server(host='0.0.0.0', port=12345):
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((host, port))
    server_socket.listen()
    print(f"Server listening on {host}:{port}")
    
    while True:
        client_socket, client_address = server_socket.accept()
        print(f"Connection from {client_address} has been established!")
        
        data = client_socket.recv(1024)
        print(f"Received data: {data.decode('utf-8')}")
        
        response = "Hello, client!"
        client_socket.sendall(response.encode('utf-8'))
        
        client_socket.close()


def start_worker(host='127.0.0.1', port=12345):
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    
    client_socket.connect((host, port))
    print(f"Connected to server at {host}:{port}")
    
    message = "Hello from {host}, server!"
    client_socket.sendall(message.encode('utf-8'))
    
    response = client_socket.recv(1024)
    print(f"Received response: {response.decode('utf-8')}")
    
    client_socket.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--main', type=str, default='')
    parser.add_argument('--port', type=str, default='')
    args = parser.parse_args()
    print(args)

    hostname = socket.gethostname()
    ip = socket.gethostbyname(hostname)

    if ip == args.main:
        start_server(ip, int(args.port))
    else:
        start_worker(args.main, int(args.port))


if __name__ == '__main__':
    print('start')
    main()