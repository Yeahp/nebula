import logging
from pathlib import Path
import json
from urllib.request import urlopen, Request


if __name__ == "__main__":
    url = "http://www.baidu.com"
    data = urlopen(url).read().decode('utf-8')
    print(data)

