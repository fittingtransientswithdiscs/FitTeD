import numpy as np
import pandas as pd
from warnings import warn
from .models import *

__all__ = ["log_window_prior"]

class log_window_prior():
    _key_parameters = []
    _repeated_parameters = []
    _early_parameters = []
    def __init__(self, bounds=None, key_parameters = ["log_mh", "a_bh", "m_disc", "r0", "tvi", "t0", "incl"], early_parameters = ["log_L", "t_decay", "log_T"]):
        '''
        Based on DataFrame object
        '''
        if type(bounds) is log_window_prior:
            key_parameters = bounds.key_parameters
            early_parameters = bounds.early_parameters
        
        
        base_dataframe = pd.DataFrame(    bounds, 
                                        columns=(key_parameters+early_parameters), 
                                        index=("lower", "upper"),
                                        dtype=float)
                                    
                                        
        base_dataframe.loc["lower", base_dataframe.loc["lower"].isna()] = -np.inf
        base_dataframe.loc["upper", base_dataframe.loc["upper"].isna()] = np.inf
        
        
        object.__setattr__(self, "_key_parameters", list(key_parameters))
        object.__setattr__(self, "_early_parameters", list(early_parameters))
        object.__setattr__(self, "_base_dataframe", base_dataframe)
        
    @property
    def key_parameters(self):
        return self._key_parameters.copy()
        
    @property
    def early_parameters(self):
        return self._early_parameters.copy()
    
    @property
    def base_dataframe(self):
        return self._base_dataframe
    
    def add_key_parameter(self, key, value=None):
        if key in self._key_parameters or key in self._early_parameters:
            raise ValueError("Parameter is already in the window")
        self.base_dataframe[key] = (value if value is not None else (-np.inf, np.inf))
        self._key_parameters.append(key)
        
    def add_early_parameter(self, key, value=None):
        if key in self._key_parameters or key in self._early_parameters:
            raise ValueError("Parameter is already in the window")
        self.base_dataframe[key] = (value if value is not None else (-np.inf, np.inf))
        self._early_parameters.append(key)
    
    def remove_parameter(self, key):
        if key in self._early_parameters:
            self._early_parameters.remove(key)
        if key in self._key_parameters:
            self._key_parameters.remove(key)
        self._base_dataframe.drop(key, axis="columns", inplace=True, errors="ignore")
    
    def as_bounds(self):
        return self.base_dataframe[self.key_parameters + self.early_parameters].T.to_numpy()
    
    def __getitem__(self, key):
        result = self.base_dataframe[key]
        if type(result) is pd.Series:
            return result
        
        return log_window_prior(result,    key_parameters = [p for p in self.key_parameters if p in result],
                                        early_parameters = [p for p in self.early_parameters if p in result])
        
    def __len__(self):
        return len(self.base_dataframe)
    
    def __getattr__(self, name):
        if name in self._key_parameters or name in self._early_parameters:
            return self._base_dataframe[name]
        return object.__getattribute__(self, name)

        
        
        
    def __setattr__(self, name, value):
        if name in ("_early_parameters", "_key_parameters", "_base_dataframe"):
            warn("Overriding these could break this.", stacklevel=2)
        try:
            object.__getattribute__(self, name)
            return object.__setattr__(self, name, value)
        except AttributeError:
            pass
                
        try:
            if name in self._key_parameters or name in self._early_parameters:
                self._base_dataframe[name] = value
                return None
        except AttributeError:
            pass
        warn("Just like pandas, you cannot create columns like this.  "
              "Please use the add_key_parameter or add_repeated_parameter functions.", stacklevel=2)
        return object.__setattr__(self, name, value)

    def __call__(self, pars):
        '''
        Log likelihood of these pars.  If any parameter is impossible, return -np.inf
        p(pars)
        '''
        
        # Check key parameters
        key_pars = pd.Series(pars[:len(self.key_parameters)],  index=self.key_parameters, dtype=float)
            
        if (key_pars > self.base_dataframe.loc["upper", self.key_parameters]).any(axis=None):
            return -np.inf
        if (key_pars < self.base_dataframe.loc["lower", self.key_parameters]).any(axis=None):
            return -np.inf
        
        early_pars = pd.Series(pars[len(self.key_parameters):],  index=self.early_parameters, dtype=float)
        if (early_pars > self.base_dataframe.loc["upper", self.early_parameters]).any(axis=None):
            return -np.inf
        if (early_pars < self.base_dataframe.loc["lower", self.early_parameters]).any(axis=None):
            return -np.inf

            
        return 0
        
    def __repr__(self):
        string = "Key parameters:       " + self._key_parameters.__repr__() + "\n"
        string += "Early parameters: " + self._early_parameters.__repr__() + "\n"
        string += "\n" + self._base_dataframe.__repr__()
        return string
    def _repr_html_(self):
        string = "<div><p>Key parameters:       " + self._key_parameters.__repr__() + "</p>"
        string += "<p>Early parameters: " + self._early_parameters.__repr__() + "</p>"
        string += self._base_dataframe._repr_html_()
        string += "</div>"
        return string        
        