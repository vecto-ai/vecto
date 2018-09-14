from zipfile import ZipFile
from requests import get
from os import path


class Downloader():
    def __init__(self):
        self.current_dir = '/'
        pass

    def unpack_archive(self, input_dir, output_dir='tmp', archive_type='.zip'):
        if archive_type == '.zip':
            with ZipFile(input_dir, '') as z:
                z.extractall(output_dir)

    def fetch_file(self, url, output_file='tmp'):
        with open(output_file, 'wb') as file:
            response = get(url)
            file.write(response.content)

    def show_dir_structure(self):
        pass

    def go_dir_up(self):
        self.current_dir = self.current_dir[:self.current_dir.rfind('/')]
        self.show_dir_structure()
        pass

    def go_dir_down(self, current_dir, subdir_name):
        self.current_dir = path.join(self.current_dir, subdir_name)
        self.show_dir_structure(current_dir)
        pass