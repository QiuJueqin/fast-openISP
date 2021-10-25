# File: basic_module.py
# Description: Basic ISP module and utilities
# Created: 2021/10/22 20:50
# Author: Qiu Jueqin (qiujueqin@gmail.com)


def register_dependent_modules(dependent_module):
    """ A decorator to register dependent ISP modules """

    if not isinstance(dependent_module, (list, tuple)):
        dependent_module = [dependent_module]

    def _register_dependent_modules(cls):
        orig_init = cls.__init__

        def override_init(self, *args, **kws):
            orig_init(self, *args, **kws)
            self.dependent_modules = tuple(dependent_module)

        cls.__init__ = override_init
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
