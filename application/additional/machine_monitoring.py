import logging
import shutil
import os
from time import sleep
from prometheus_client import Info, Gauge
import pkg_resources
import psutil

from application.config import DATA_FOLDER, DATA_FORMAT_FILE, DATA_PIPELINE_FILE

i_stor = Gauge('machine_capabilities_free_storage',
               'Information on current machine capabilities as seen by FL LO (memory and storage in GB)')
i_mem = Gauge('machine_capabilities_free_memory',
              'Information on current machine capabilities as seen by FL LO (memory and storage in GB)')
i_gpu = Gauge('machine_capabilities_gpu',
              'Information on current machine capabilities as seen by FL LO (memory and storage in GB)')


def check_machine_capabilities():
    """                    
    Update the values of Prometheus metrics to reflect the current state of available capabilities.
    """
    i_stor.set(check_storage())
    i_mem.set(check_memory())
    i_gpu.set(check_gpu())


def check_data_changes(folder=DATA_FOLDER):
    """
    Checks when was the last time the data in the folder has been modified
    and so the preprocessed cache folder should be reloaded.
    Modification here means that either the format file,
    or the local pipeline configuration has been modified
    or the number of files in the directory has changed.
    """
    time_folder = os.path.getmtime(folder)
    time_format = os.path.getmtime(os.path.join(folder, DATA_FORMAT_FILE)) if os.path.isfile(
        os.path.join(folder, DATA_FORMAT_FILE)) else None
    time_pipeline = os.path.getmtime(os.path.join(folder, DATA_PIPELINE_FILE)) if os.path.isfile(
        os.path.join(folder, DATA_PIPELINE_FILE)) else None
    return time_folder, time_format, time_pipeline


def setup_check_data_changes(timestep=5):
    """
    Is responsible for regularly checking if the time of the last folder
    modification has changed and the data transformation caching should
    therefore be renewed.
    """
    logger = logging.getLogger()
    # log all messages, debug and up
    logger.setLevel(logging.INFO)
    logging.log(logging.INFO, "Start checking for data changes")

    curr_time_folder, curr_time_format, curr_time_pipeline = check_data_changes()
    while True:
        check_machine_capabilities()
        new_t_folder, new_t_format, new_t_pipeline = check_data_changes()
        if curr_time_folder != new_t_folder or curr_time_format != new_t_format or curr_time_pipeline != new_t_pipeline:
            logging.log(logging.INFO, "The folder has been modifed")
            curr_time_folder = new_t_folder
            curr_time_pipeline = new_t_pipeline
            curr_time_format = new_t_format
        sleep(timestep)


def check_storage():
    """Checks how much storage is left on the device"""
    _, _, free = shutil.disk_usage("/")
    return free // (2 ** 30)


def check_memory():
    """Checks how much memory is left on the device"""
    memory = psutil.virtual_memory()
    free = memory.free
    return free // 1000000000


def check_gpu():
    """Checks if cuda available and assumes it's not if torch is also not there"""
    try:
        import torch
    except ImportError:
        return False
    else:
        return torch.cuda.is_available()


def check_packages():
    """"Checks out the dict of installed packages along with versions"""
    package_names = [d.project_name for d in pkg_resources.working_set]
    return {key: pkg_resources.get_distribution(key).version for key in package_names}


def check_models():
    """When we decide to store specific kinds of data in the local database
    We'll define their retrieval here"""
    return {}
