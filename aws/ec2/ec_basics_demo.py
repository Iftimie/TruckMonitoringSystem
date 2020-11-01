# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Purpose
Shows how to use the AWS SDK for Python (Boto3) with the Amazon Elastic Compute Cloud
(Amazon EC2) API to create an instance, perform some management tasks on the instance,
and clean up everything created during the demo.
"""

import logging
import pprint
import time
import urllib.request
import boto3

import ec2_setup
import ec2_instance_management
import ec2_teardown
from botocore.exceptions import ClientError
import os
logger = logging.getLogger(__name__)
ec2 = boto3.resource('ec2')

logger = logging.getLogger(__name__)


def make_unique_name(name):
    return f'demo-ec2-{name}-{time.time()}'


def setup_demo(current_ip_address, ami_image_id, key_file_name):
    """
    Sets up prerequisites and creates instances used in the demo.
    When this function returns, the instances are running and ready to use.
    :param current_ip_address: The public IP address of the current computer.
    :param ami_image_id: The Amazon Machine Image (AMI) that is used to create the
                         instances for the demo.
    :param key_file_name: The name of a local file that contains the private key
                          that is used to connect to the instances using SSH.
    :return: The newly created instances, security groups, key pair, and
             Elastic IP object.
    """
    key_pair = ec2_setup.create_key_pair(make_unique_name('key'), key_file_name)

    ssh_sec_group = ec2_setup.setup_security_group(
        make_unique_name('ssh-group'),
        f'Demo group that allows SSH from {current_ip_address}.',
        current_ip_address)

    ssh_instance_broker = ec2_setup.create_instance(
        ami_image_id, 't2.micro', key_pair.key_name, [ssh_sec_group.group_name])

    ssh_instance_worker = ec2_setup.create_instance(
        ami_image_id, 't2.micro', key_pair.key_name, [ssh_sec_group.group_name])

    print(f"Waiting for instances to start...")
    ssh_instance_broker.wait_until_running()
    ssh_instance_worker.wait_until_running()

    return (ssh_instance_broker, ssh_instance_worker), (ssh_sec_group, ), key_pair


def management_demo(ssh_instance_broker, ssh_instance_worker, key_file_name):
    """
    Shows how to perform management actions on an Amazon EC2 instance.
    * Associate an Elastic IP address with an instance.
    * Stop and start an instance.
    * Allow one instance to connect to another by setting an inbound rule
      in the target instance's security group that allows traffic from the
      source instance's security group.
    * Change an instance's security group to another security group.
    :param ssh_instance: An instance that is associated with a security group that
                         allows access from this computer using SSH.
    :param no_ssh_instance: An instance that is associated with a security group
                            that does not allow access using SSH.
    :param key_file_name: The name of a local file that contains the private key
                          for the demonstration instances.
    """
    ssh_instance_broker.load()
    ssh_instance_worker.load()

    print(f"At this point, you can SSH to broker {ssh_instance_broker.instance_id} "
          f"at another command prompt by running")
    print(f"\tssh -i {key_file_name} ubuntu@{ssh_instance_broker.public_ip_address}")

    print(f"At this point, you can SSH to worker {ssh_instance_worker.instance_id} "
          f"at another command prompt by running")
    print(f"\tssh -i {key_file_name} ubuntu@{ssh_instance_worker.public_ip_address}")

    os.system('chmod 400 demo-key-file.pem')
    with open('ip_broker.txt', 'w') as f:
        f.write(ssh_instance_broker.public_ip_address)
    with open('ip_worker.txt', 'w') as f:
        f.write(ssh_instance_worker.public_ip_address)
    input("Press Enter when you're ready to continue the demo.")


def teardown_demo(instances, security_groups, key_pair, key_file_name):
    """
    Cleans up all resources created during the demo, including terminating the
    demo instances.
    After an instance is terminated, it persists in the list
    of instances in your account for up to an hour before it is ultimately removed.
    :param instances: The demo instances to terminate.
    :param security_groups: The security groups to delete.
    :param key_pair: The security key pair to delete.
    :param key_file_name: The private key file to delete.
    :param elastic_ip: The Elastic IP to release.
    """
    for instance in instances:
        ec2_teardown.terminate_instance(instance.instance_id)
        instance.wait_until_terminated()
    print("Terminated the demo instances.")

    for security_group in security_groups:
        ec2_teardown.delete_security_group(security_group.group_id)
    print("Deleted the demo security groups.")

    ec2_teardown.delete_key_pair(key_pair.name, key_file_name)
    print("Deleted demo key.")


def get_ami_image_id():
    ami_image_id = "ami-05c424d59413a2876"
    return ami_image_id


def run_demos():
    """
    """
    current_ip_address = urllib.request.urlopen('http://checkip.amazonaws.com')\
        .read().decode('utf-8').strip()

    print("All ec2 instances: ")
    for i in ec2.instances.all():
        print(i)

    key_file_name = 'demo-key-file.pem'
    if os.path.exists(key_file_name):
        os.remove(key_file_name)
    instances, security_groups, key_pair = setup_demo(
        current_ip_address, get_ami_image_id(), key_file_name)
    management_demo(*instances, key_file_name)

    print("All ec2 instances: ")
    for i in ec2.instances.all():
        print(i)

    teardown_demo(instances, security_groups, key_pair, key_file_name)


if __name__ == '__main__':
    run_demos()