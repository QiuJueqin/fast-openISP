# File: basic_module.py
# Description: Basic ISP module and utilities
# Created: 2021/10/22 20:50
# Author: Qiu Jueqin (qiujueqin@gmail.com)


def register_dependent_modules(dependent_module_names):
    """ A decorator to register dependent ISP modules """

    if not isinstance(dependent_module_names, (list, tuple)):
        dependent_module_names = [dependent_module_names]

    def _register_dependent_modules(cls):
        orig_init = cls.__init__

        def __init__(self, *args, **kws):
            orig_init(self, *args, **kws)
            self.register_dependent_modules = tuple(dependent_module_names)

        cls.__init__ = __init__
        return cls

    return _register_dependent_modules


class BasicModule:
    def __init__(self, cfg):
        self.cfg = cfg

        module_name = self.__class__.__name__.lower()
        self.params = cfg[module_name] if module_name in cfg else None

    def execute(self, data):
        """
        :param data: a dict containing data flow in the pipeline, as well as other intermediate results,
            e.g., YCbCr image from color space conversion module, edge map from edge enhancement module.
            Instead of returning a processed result, the execute() method in each module will in-place
            modify the data dict
        """
        raise NotImplemented
