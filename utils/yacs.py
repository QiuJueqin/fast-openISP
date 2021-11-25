# File: yacs.py
# Description: Core module for YACS
# Created: 2021/6/15 20:30
# Author: Qiu Jueqin (qiujueqin@gmail.com)


import os.path as op
import copy
import argparse
import pathlib
from collections import OrderedDict, defaultdict
from contextlib import contextmanager

import yaml

"""
Yet Another Configuration System: a lightweight yet sufficiently powerful 
configuration system with no 3rd-party dependency

You can instantiate a Config object from a yaml file:

    >>> cfg = Config('your_config.yaml')

or from a regular (optionally nested) dict:

    >>> dic = {'train': {'optimizer': 'adam', 'lr': 0.001}}
    >>> cfg = Config(dic)

or from a Namespace object created by the argparse package:

    >>> import argparse
    >>> parser = argparse.ArgumentParser()
    >>> parser.add_argument('--batch_size', type=int, default=128)
    >>> cfg = Config(parser.parse_args())

or/and by merging from others:

    >>> cfg = Config('your_config.yaml')
    >>> cfg.merge('another_config.yaml')

Config supports two ways to access its containing attributes:

The regular dict way:

    >>> bs = cfg['batch_size']

and the dotted-dict way (more recommended since it requires less keyboard 
hits and save your line width)

    >>> bs = cfg.batch_size

See README.md and ./examples directory for more usage hints.
"""


class Config(OrderedDict):

    def __init__(self, init=None):
        """
        :param init: dict | yaml filepath | argparse.Namespace
        """

        self.__dict__['__immutable__'] = False

        if init is None:
            super().__init__()
        elif isinstance(init, dict):
            self.from_dict(init)
        elif isinstance(init, str):
            self.from_yaml(init)
        elif isinstance(init, argparse.Namespace):
            self.from_namespace(init)
        else:
            raise TypeError(
                f'Config could only be instantiated from a dict, a yaml '
                f'filepath, or an argparse.Namespace object, but given a '
                f'{type(init)} object'
            )

    # ---------------- Immutability ----------------

    @property
    def is_frozen(self):
        return self.__dict__['__immutable__']

    def freeze(self):
        self._set_immutable(True)

    @contextmanager
    def unfreeze(self):
        """
        When a Config is frozen (a default action once it is instantiated),
        users have to use the unfreeze() context manager to modify it:

        >>> cfg = Config('default_config.yaml')
        >>> with cfg.unfreeze():
        >>>     cfg.batch_size = 512
        """

        try:
            self._set_immutable(False)
            yield self
        finally:
            self.freeze()

    def _set_immutable(self, is_immutable):
        """ Recursively set immutability. """

        self.__dict__['__immutable__'] = is_immutable
        for v in self.values():
            if isinstance(v, Config):
                v._set_immutable(is_immutable)

    # ---------------- Set & Get ----------------

    def __setattr__(self, key, value):
        if self.is_frozen:
            raise AttributeError('attempted to modify an immutable Config')
        self[key] = value

    def __setitem__(self, key, value):
        if self.is_frozen:
            raise AttributeError('attempted to modify an immutable Config')
        super().__setitem__(key, value)

    def __getattr__(self, key):
        if key in self:
            return self[key]
        else:
            raise AttributeError(
                f'attempted to access a non-existing attribute: {key}'
            )

    # ---------------- Input ----------------

    def from_dict(self, dic):
        """
        Instantiation from a regular dict. If the dict is nested, it will
        create children Config objects recursively.
        """

        if not isinstance(dic, dict):
            raise TypeError(f'expected a dict, but given a {type(dic)}')

        super().__init__(Config._from_dict(dic))
        self.freeze()

    def from_yaml(self, yaml_path):
        """ Instantiation from a yaml file. """

        if not isinstance(yaml_path, (str, pathlib.Path)):
            raise TypeError(
                f'expected a path string or a pathlib.Path object, but given '
                f'a {type(yaml_path)}'
            )

        if not op.isfile(str(yaml_path)):
            raise FileNotFoundError(f'file {yaml_path} does not exist')

        with open(yaml_path, 'r') as fp:
            dic = yaml.safe_load(fp)

        super().__init__(Config._from_dict(dic))
        self.freeze()

    def from_namespace(self, namespace):
        """
        Instantiation from an argparse.Namespace object.

        Since argparse doesn't support nested arguments, we treat dot
        separator in the arguments as a notation to recursively create a
        child Config object.

        For example, creating an argparse.ArgumentParser with '--foo.bar'
        argument:

        >>> import argparse
        >>> parser = argparse.ArgumentParser()
        >>> parser.add_argument('--foo.bar', type=int, default=42)
        >>> args = parser.parse_args()  # an argparse.Namespace object

        Given the returned argparse.Namespace object 'args', from_namespace()
        will create a Config object as if it was instantiated from a nested
        dict d = {'foo': {'bar': 42}}
        """

        if not isinstance(namespace, argparse.Namespace):
            raise TypeError(
                f'expected an argparse.Namespace object, but given a '
                f'{type(namespace)} '
            )

        nested_dict = self._separator_dict_to_nested_dict(vars(namespace))
        super().__init__(Config._from_dict(nested_dict))
        self.freeze()

    def merge(self, other, allow_new_attr=False, keep_existed_attr=True):
        """
        Recursively merge from other object

        :param other: Config object | dict | yaml filepath |
            argparse.Namespace object
        :param allow_new_attr: whether allow to add new attributes

        Example:

        >>> cfg = Config({'optimizer': 'adam'})
        >>> cfg.merge({'lr': 0.001}, allow_new_attr=True)
        >>> cfg.print()

        optimizer: adam
        lr: 0.001

        >>> cfg.merge({'weight_decay': 1E-7}, allow_new_attr=False)

        AttributeError: attempted to add a new attribute: weight_decay

        :param keep_existed_attr: whether keep those attributes that are not
            in 'other'. You may wish to trigger this if requires to completely
            replace a child Config object. See example/examples.py: Example 5
            for a practical usage.

        Example:

        >>> cfg1 = Config({'foo': {'Alice': 0, 'Bob': 1}})
        >>> cfg2 = cfg1.copy()
        >>> another = {'foo': {'Carol': 42}}
        >>> cfg1.merge(another, allow_new_attr=True)
        >>> cfg1.print()

        foo:
            Alice: 0
            Bob: 1
            Carol: 42

        >>> cfg2.merge(another, allow_new_attr=True, keep_existed_attr=False)
        >>> cfg2.print()

        foo:
            Carol: 42
        """

        if isinstance(other, Config):
            pass
        elif isinstance(other, (dict, str, pathlib.Path, argparse.Namespace)):
            other = Config(other)
        else:
            raise TypeError(
                f'attempted to merge from an unsupported {type(other)} object'
            )

        def _merge(source_cfg, other_cfg, add_new, keep_existed):
            """ Recursively merge the new Config object into the source one """

            with source_cfg.unfreeze():
                for k, v in other_cfg.items():
                    if k not in source_cfg and not add_new:
                        raise AttributeError(
                            f'attempted to add an attribute {k} but it is not '
                            f'found in the source Config. Set `allow_new_attr` '
                            f'to True if requires to add new attributes'
                        )

                    if isinstance(v, Config):
                        if isinstance(source_cfg.get(k, None), Config):
                            _merge(source_cfg[k], v, add_new, keep_existed)
                        else:
                            source_cfg[k] = v
                    else:
                        source_cfg[k] = copy.deepcopy(v)

                if not keep_existed:
                    source_keys = list(source_cfg.keys())
                    for k in source_keys:
                        if k not in other_cfg and \
                                not isinstance(source_cfg[k], Config):
                            source_cfg.remove(k)

        _merge(self, other, allow_new_attr, keep_existed_attr)

    # ---------------- Output ----------------

    def to_dict(self, alphabetical=False):
        """
        Convert a Config object to a (nested) regular dict.
        An inverse method to self.from_dict()
        """

        def _to_dict(config):
            if alphabetical:
                config = sorted(config.items(), key=lambda x: x[0])
            dic = dict(config)
            for k, v in dic.items():
                if isinstance(v, Config):
                    dic[k] = _to_dict(v)
            return dic

        return _to_dict(self)

    def to_parser(self):
        """
        Create an argparse.ArgumentParser object with keys as arguments.

        Since argparse doesn't support nested arguments, we concatenate keys
        over hierarchies into a *dotted* argument. For example, supposing a
        Config object is organized as following:

        >>> cfg = {
        >>>     'foo': {
        >>>         'bar': 42
        >>>     }
        >>> }

        Then keys 'foo' and 'bar' will be concatenated into a new argument
        '--foo.bar', so by calling cfg.to_parser(), it is equivalent to
        creating an ArgumentParser object as following:

        >>> parser = argparse.ArgumentParser()
        >>> parser.add_argument('--foo.bar', type=int, default=42)
        >>> return parser

        :return: an argparse.ArgumentParser object
        """
        separator_dict = self._nested_dict_to_separator_dict(self.to_dict())

        parser = argparse.ArgumentParser()
        for k, v in separator_dict.items():
            parser.add_argument(f'--{k}', type=type(v), default=v)

        return parser

    def dump(self, save_path):
        """ Dump a Config object into a yaml file """
        with open(save_path, 'w') as fp:
            yaml.safe_dump(self.to_dict(alphabetical=True), fp)

    def copy(self):
        """ Create a deep copy of the Config object """
        return Config(copy.deepcopy(self.to_dict()))

    # ---------------- Misc ----------------

    def __repr__(self):
        return self.to_dict().__repr__()

    def __str__(self):
        return self.to_dict().__repr__()

    def _format(self):

        def _to_string(dic, indent=0):
            texts = []
            for k, v in dic.items():
                if not isinstance(v, Config):
                    texts += ['{:<25}{}'.format(' ' * indent + k + ':', str(v))]
                else:
                    texts += [k + ':'] + _to_string(v, indent=indent + 2)
            return texts

        return '\n'.join(_to_string(self))

    def print(self, streamer=print):
        streamer(self._format())

    def remove(self, key):
        """ Remove an attribute by its key. """

        if self.is_frozen:
            raise AttributeError('attempted to modify an immutable Config')
        if key not in self:
            raise AttributeError(
                f'attempted to delete a non-existing attribute: {key}'
            )

        del self[key]

    # ---------------- Private ----------------

    @classmethod
    def _from_dict(cls, dic):
        dic = copy.deepcopy(OrderedDict(dic))
        for k, v in dic.items():
            if isinstance(v, dict):
                dic[k] = cls(v)
            elif isinstance(v, list):  # load list as tuple for safety
                dic[k] = tuple(cls(x) if isinstance(x, dict) else x for x in v)

        return dic

    @staticmethod
    def _separator_dict_to_nested_dict(separator_dict, separator='.'):
        """
        Create a nested dict from a single-hierarchy dict.

        For example, given a non-nested dict in which part of keys contains
        dots:

        d = {
            'foo': 42,
            'alpha.beta': 123,
            'bar.baz.hello': 'HELLO',
            'bar.baz.world': 'WORLD'
        }

        _separator_dict_to_nested_dict(d, separator='.') converts it into a
        nested dict:

        {
            'foo': 42,
            'alpha': {'beta': 123},
            'bar': {
                'baz': {
                    'hello': 'HELLO',
                    'world': 'WORLD'
                }
            }
        }

        :param separator_dict: a non-nested dict
        :return: a nested dict
        """
        separator_dict = copy.deepcopy(separator_dict)

        def _init_nested_dict():
            return defaultdict(_init_nested_dict)

        def _default_to_dict(d):
            """ Convert a defaultdict object into a regular dict """
            if isinstance(d, defaultdict):
                d = {kk: _default_to_dict(vv) for kk, vv in d.items()}
            return d

        nested_dict = _init_nested_dict()

        for k, v in separator_dict.items():
            tmp_d = nested_dict
            keys = k.split(separator)
            for sub_key in keys[:-1]:
                tmp_d = tmp_d[sub_key]
            tmp_d[keys[-1]] = v

        return _default_to_dict(nested_dict)

    @staticmethod
    def _nested_dict_to_separator_dict(nested_dict, separator='.'):
        """
        Create a single-hierarchy dict from a nested dict.

        For example, given a nested dict:

        d = {
            'foo': 42,
            'alpha': {'beta': 123},
            'bar': {
                'baz': {
                    'hello': 'HELLO',
                    'world': 'WORLD'
                }
            }
        }

        _nested_dict_to_separator_dict(d, separator='.') converts it into a
        non-nested dict in which each key is the concatenation of keys from
        different hierarchies:

        {
            'foo': 42,
            'alpha.beta': 123,
            'bar.baz.hello': 'HELLO',
            'bar.baz.world': 'WORLD'
        }

        :param nested_dict: a regular (optionally nested) dict
        :return: a non-nested dict whose keys contain separators
        """
        nested_dict = copy.deepcopy(nested_dict)

        def _create_separator_dict(x, key='', separator_dict={}):
            if isinstance(x, dict):
                for k, v in x.items():
                    kk = f'{key}{separator}{k}' if key else k
                    _create_separator_dict(x[k], kk)
            else:
                separator_dict[key] = x
            return separator_dict

        return _create_separator_dict(nested_dict)
