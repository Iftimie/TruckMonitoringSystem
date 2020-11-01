import paramiko
import traceback
import os
# in case I want interactive
# https://stackoverflow.com/questions/6203653/how-do-you-execute-multiple-commands-in-a-single-session-in-paramiko-python/6203877#6203877
# https://stackoverflow.com/questions/8932862/how-do-i-change-directories-using-paramiko
# https://stackoverflow.com/questions/34559158/paramiko-1-16-0-readlines-decode-error

# http://cardcode121.blogspot.com/2012/08/python-paramiko-1160-readlines-decode.html
def u(s, encoding='utf8'):
    """cast bytes or unicode unicode"""
    if isinstance(s, bytes):
        try:
            return s.decode(encoding)
        except UnicodeDecodeError:
            return s.decode('iso-8859-1')
    elif isinstance(s, str):
        return s
    else:
        raise typeerror("expected unicode or bytes, got %r" % s)

paramiko.py3compat.u = u

def run_commands(c, command_list):
    for command in command_list:
        stdin, stdout, stderr = c.exec_command(command, get_pty=True)
        # stdout._set_mode('b')
        for line in iter(lambda: stdout.readline(2048), ""):
            print(line, end='')
        for line in stderr.readlines():
            print(line, end='')
        exit_status = stdout.channel.recv_exit_status()
        if exit_status == 0:
            print("Command finished")
        else:
            print("Error in command", exit_status)

def ssh_login():
    try:
        cert = paramiko.RSAKey.from_private_key_file('demo-key-file.pem')
        c = paramiko.SSHClient()
        c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        with open('ip_broker.txt', 'r') as f:
            ip_broker = f.read()
        c.connect(hostname=ip_broker, username='ubuntu', pkey=cert)
        run_commands(c, ['git clone https://github.com/Iftimie/TruckMonitoringSystem.git',
                         'cd TruckMonitoringSystem; git checkout develop',
                         'cd TruckMonitoringSystem; git pull',
                         'cd TruckMonitoringSystem; ls',
                         'cd TruckMonitoringSystem; sudo bash install_deps_host.sh',
                         'cd TruckMonitoringSystem; sudo bash run_broker.sh'])
        c.close()

        with open('ip_worker.txt', 'r') as f:
            ip_worker = f.read()
        c.connect(hostname=ip_worker, username='ubuntu', pkey=cert)
        run_commands(c, ['git clone https://github.com/Iftimie/TruckMonitoringSystem.git',
                         'cd TruckMonitoringSystem; git checkout develop',
                         'cd TruckMonitoringSystem; git pull',
                         'cd TruckMonitoringSystem; ls',
                         'echo "'+ip_broker+':5002" > TruckMonitoringSystem/discovery.txt',
                         'cd TruckMonitoringSystem; ls',
                         'cat TruckMonitoringSystem/discovery.txt',
                         'cd TruckMonitoringSystem; sudo bash install_deps_host.sh',
                         'sudo docker network create broker_mynet', # todo I have to fix this
                         'cd TruckMonitoringSystem; sudo bash run_worker.sh'])
        c.close()

    except Exception as e:
        traceback.print_exc()
        print("Connection Failed!!!")

ssh_login()