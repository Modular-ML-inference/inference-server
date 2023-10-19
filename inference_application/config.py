import os


def if_env(keyword):
    if keyword in os.environ:
        return os.environ[keyword]
    else:
        return ''


REPOSITORY_ADDRESS = if_env('REPOSITORY_ADDRESS')
