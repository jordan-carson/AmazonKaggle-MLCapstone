from abc import ABC, abstractmethod
import pandas


class AbstractBaseClass(ABC):

    @abstractmethod
    def read_file(self, full_file_path):
        try:
            return pandas.read_csv(full_file_path)
        except Exception as err:
            print("Read file method requires a csv." + '\n'
                  + 'Error: ' + str(err))

    @abstractmethod
    def import_image(self, image_path):
        print("Reading a image file from " + image_path)


# https://www.python-course.eu/python3_object_oriented_programming.php
