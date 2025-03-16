from scapy.all import ARP, Ether, sr
import ipaddress

def scan_network(ip_range):
    # Формируем ARP-запрос для указанного диапазона
    arp_request = ARP(pdst=str(ip_range))
    broadcast = Ether(dst="ff:ff:ff:ff:ff:ff")
    packet = broadcast / arp_request

    # Отправляем пакет и получаем ответы
    result = sr(packet, timeout=2, verbose=0)[0]

    # Парсим и выводим информацию о найденных устройствах
    devices = []
    for sent, received in result:
        devices.append({'IP': received.psrc, 'MAC': received.hwsrc})

    # Выводим результат
    print("Найденные устройства:")
    for device in devices:
        print(f"IP: {device['IP']}, MAC: {device['MAC']}")

# Укажите диапазон IP-адресов для сканирования
network = "192.168.0.0/24"  # Замените на ваш диапазон

# Сканируем сеть
scan_network(ipaddress.ip_network(network, strict=False))
