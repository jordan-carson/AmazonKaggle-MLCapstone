from abc import ABC, abstractmethod
import pandas
import datetime


class AbstractBaseClass(ABC):

    @abstractmethod
    def read_file(self, full_file_path):
        return

    @abstractmethod
    def import_image(self, image_path):
        return

    @abstractmethod
    def write_file(self, folder_path):
        return
    # https://www.python-course.eu/python3_object_oriented_programming.php


class SumList(object):
    def __init__(self, this_list):
        self.mylist = this_list

    def __add__(self, other):
        new_list = [x + y for x, y in zip(self.mylist, other.mylist)]

        return SumList(new_list)

    def __repr__(self):
        return str(self.mylist)


class ReadImageData(AbstractBaseClass):

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


class WriteFile:

    def __init__(self, filename, writer):
        self.fh = open(filename, 'w')
        self.formatter = writer()

    def write(self, text):
        self.fh.write(self.formatter.format(text))
        self.fh.write('\n')

    def close(self):
        self.fh.close()


class CSVFormatter:
    """module to format csv output"""
    def __init__(self):
        self.delim = ','

    def format(self, this_list):
        new_list = []
        for element in this_list:
            if self.delim in element:
                new_list.append('"{0}'.format(element))
            else:
                new_list.append(element)
        return self.delim.join(new_list)


class LogFormatter:
    """For formating log files"""
    def format(self, this_line):
        dt = datetime.datetime.now()
        date_str = dt.strftime('%Y-%m-%d %H:%M')
        return '{0}  {1}'.format(date_str, this_line)







































