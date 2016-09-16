# -*- coding: utf-8 -*-


class MyLogger(object):

    def __init__(self, path):
        self.f = open(path, "w")

    def write(self, msg):
        print msg
        self.f.write("%s\n" % msg)
        self.f.flush()

    def close(self):
        self.f.close()
