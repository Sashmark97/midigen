import os
import abc
import yaml
import torch
from torch import nn
from abc import ABC, abstractmethod


class AbstractYAMLMeta(yaml.YAMLObjectMetaclass, abc.ABCMeta):
    """
    Metaclass used to fix conflicts in multiple inheritance.
    """

    def __init__(cls, name, bases, kwds):
        super().__init__(name, bases, kwds)
        cls.yaml_tag = f"!{cls.__name__}"
        cls.yaml_loader.add_constructor(f"!{cls.__name__}", cls.from_yaml)
        cls.yaml_dumper.add_representer(cls, cls.to_yaml)

class Model(ABC, yaml.YAMLObject, nn.Module, metaclass=AbstractYAMLMeta):
    """Core class for deep learning models"""
    yaml_loader = yaml.Loader
    yaml_dumper = yaml.Dumper

    @classmethod
    def from_yaml(cls, loader, node):
        data = loader.construct_mapping(node, deep=True)
        return cls(**data)

    @classmethod
    def to_yaml(cls, dumper, node):
        return dumper.represent_mapping(cls.yaml_tag,
                                        node.get_parameters())

    def save(self, save_path):
        """
        Saves model to specified file.

        Parameters
        ----------
        save_path : str
            Path to the save file of the model.

        """
        folder_path = os.path.dirname(save_path)
        if len(folder_path) > 0 and not os.path.exists(folder_path):
            os.makedirs(folder_path)

        torch.save({
            "model_state_dict": self.state_dict(),
            "parameters": self.get_parameters()
        }, save_path)

    @classmethod
    def load(cls, load_path, device='cpu'):
        """
        Loads model to the specified device.

        Parameters
        ----------
        load_path : str
            Path to model save file.
        device : str, optional (default='cpu')
            Device to load model to.

        Returns
        -------
        :class:'~domadkd.core.Model'
            Object of :class:'~domadkd.core.Model' class.
        """
        checkpoint = torch.load(load_path, map_location=device)
        parameters = checkpoint["parameters"]
        model = cls(**parameters).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        return model

    @abstractmethod
    def get_parameters(self):
        """
        Returns all model parameters for initialization.

        Returns
        -------
        dict
            Model parameters dictionary.

        """
        pass
