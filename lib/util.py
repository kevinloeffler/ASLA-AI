from datetime import datetime, timedelta

import requests
import ssl


def create_unverified_ssl_context():
    """create unverified ssl context to download models"""
    requests.packages.urllib3.disable_warnings()
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context


class Timer:
    def __init__(self):
        self.start_time: datetime | None = None

    def start(self):
        self.start_time = datetime.now()

    def stop(self, as_string: bool = False) -> timedelta | str:
        if not self.start_time:
            raise UnboundLocalError('Timer needs to be started before it can be stopped')
        duration = datetime.now() - self.start_time
        self.start_time = None
        if not as_string:
            return duration
        return f'{duration.seconds}:{duration.microseconds}'

