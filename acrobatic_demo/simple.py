import logging
import os
from time_prac import time_prac
from acrobatic_demo.download import setup_download_dir, get_links, download_link
# or we may take relative call:
# from .Download import setup_download_dir, get_links, download_link

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logging.getLogger('requests').setLevel(logging.CRITICAL)
logger = logging.getLogger(__name__)


def main():
    ts = time_prac()
    client_id = os.getenv('IMGUR_CLIENT_ID')
    if not client_id:
        raise Exception("Couldn't find IMGUR_CLIENT_ID environment variable!")
    download_dir = setup_download_dir()
    links = [l for l in get_links(client_id) if l.endswith('.jpg')]
    for link in links:
        download_link(download_dir, link)
    print('Took {}s'.format(time_prac() - ts))


if __name__ == '__main__':
    main()
