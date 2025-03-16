import modbus_tk.modbus_tcp as modbus_tcp
import modbus_tk.defines as cst

try:
    # Создаём TCP-клиент
    client = modbus_tcp.TcpMaster(host='192.168.1.39', port=502)
    client.set_timeout(10.0)

    # Чтение регистров для проверки до записи
    response = client.execute(1, cst.READ_HOLDING_REGISTERS, 0, 5)
    print("Данные регистров до записи:", response)

    # Запись нулей в первые 5 регистров
    values_to_write = [0, 0, 0, 0, 0]
    client.execute(1, cst.WRITE_MULTIPLE_REGISTERS, 0, output_value=values_to_write)
    print("Запись успешна:", values_to_write)

    # Чтение регистров для проверки после записи
    response = client.execute(1, cst.READ_HOLDING_REGISTERS, 0, 5)
    print("Данные регистров после записи:", response)

except Exception as e:
    print("Ошибка:", e)

