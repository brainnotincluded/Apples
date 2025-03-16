import socket
import json

def tcp_server(host='0.0.0.0', port=8000):
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        server_socket.bind((host, port))
        server_socket.listen(5)
        print(f"Сервер запущен на {host}:{port}. Ожидание соединений...")

        while True:
            client_socket, client_address = server_socket.accept()
            print(f"Подключился клиент: {client_address}")
            while True:
                try:
                    data = json.loads(client_socket.recv(1024).decode('utf-8'))
                except:
                    data = None
                if not data:
                    break
                print(f"{client_address}: {data['cup_number']}")
                # print(data.keys())
            print(f"Соединение с {client_address} закрыто")
            client_socket.close()
    except Exception as e:
        print(f"Произошла ошибка: {e}")
    finally:
        server_socket.close()

if __name__ == "__main__":
    tcp_server()
