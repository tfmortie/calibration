""" 
Some important transformers and other help functions for working with the calibration module.

Author: Thomas Mortier
Date: March 2022


TODO: 
    - argument checks
"""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import column_or_1d
from sklearn.utils.validation import check_is_fitted, check_random_state

class Transformer(TransformerMixin, BaseEstimator):
    """ Transformer which represents a convenience wrapper for all transformers 
    needed to work with the calibration module.
    
    Parameters
    ----------
    k : tuple of int, default=None
        Min and max number of children a node can have in the random generated tree. Hierarchical labels
        are assumed to be given when set to None.
    sep : str, default=';'
        String used for path encodings.
    random_state : RandomState or an int seed, default=None
        A random number generator instance to define the state of the
        random permutations generator.
    
    Attributes
    ----------
    k : tuple of int
        Represents min and max number of children a node can have in the random generated tree.
    sep : str
        String used for path encodings.
    random_state_ : RandomState or an int seed
        A random number generator instance to define the state of the
        random permutations generator.
    hlt : FHLabelTransformer
        Flat to hierarchical label transformer.
    hle : HFLabelTransformer
        Hierarchical to flat label transformer.
    hstruct_ : list
        List BFS structure which represents the hierarchy in terms of encoded labels after fit.
    classes_ : list 
        Classes (original) seen during fit.
    """
    def __init__(self, k=None, sep=';', random_state=None):
        self.k = k
        self.sep = sep
        self.random_state = random_state
        if k is not None:
            self.hlt = FHLabelTransformer(self.k, self.sep, random_state=self.random_state)
        else:
            self.hlt = None
        self.hle = HFLabelTransformer(sep=";")

    def fit(self, y):
        """ Fit transformer.
        
        Parameters
        ----------
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        self : returns an instance of self.
        """
        # store classes seen during fit
        self.classes_ = list(np.unique(y))
        if self.hlt is not None:
            self.hlt = self.hlt.fit(y)
            self.hle = self.hle.fit(self.hlt.transform(y))
        else:
            self.hle = self.hle.fit(y)
        # store the hierarchy
        self.hstruct_ = self.hle.hstruct_

        return self

    def fit_transform(self, y, path=False):
        """ Fit transformer return encoded labels.

        Parameters
        ----------
        y : array-like of shape (n_samples,)
            Target values.
        path : boolean, default=False
            Whether paths need to be returned or encoded flat labels.

        Returns
        -------
        y_transformed : array-like of shape (n_samples,)
        """
        self.fit(y)
        y_transformed = self.transform(y, path)

        return y_transformed

    def transform(self, y, path=True):
        """ Transform labels.

        Parameters
        ----------
        y : array-like of shape (n_samples,)
            Target values.
        path : boolean, default=False
            Whether paths need to be returned or encoded flat labels.

        Returns
        -------
        y_transformed : array-like of shape (n_samples,)
        """
        if self.hlt is not None:
            y = self.hlt.transform(y)
        y_transformed = self.hle.transform(y, path)

        return y_transformed

    def inverse_transform(self, y):
        """ Inverse transform labels. 

        Parameters
        ----------
        y : ndarray of shape (n_samples,)
            Target values.

        Returns
        -------
        y_transformed : ndarray of shape (n_samples,)
        """
        y_l = [[l] for l in list(y)]
        y_transformed = self.hle.inverse_transform(y_l)
        if self.hlt is not None:
            y_transformed = self.hlt.inverse_transform(y_transformed)

        return y_transformed

class FHLabelTransformer(TransformerMixin, BaseEstimator):
    """ Flat to hierarchical label transformer where a hierarchy is generated by some random k-ary
    tree.

    Parameters
    ----------
    k : tuple of int, default=(2,2)
        Min and max number of children a node can have in the random generated tree.     
    sep : str, default=';'
        String used for path encodings.
    random_state : RandomState or an int seed, default=None
        A random number generator instance to define the state of the
        random permutations generator.
    
    Attributes
    ----------
    k : tuple of int
        Represents min and max number of children a node can have in the random generated tree.
    sep : str
        String used for path encodings.
    random_state_ : RandomState or an int seed
        A random number generator instance to define the state of the
        random permutations generator.
    classes_ : list 
        Classes (original) seen during fit.
    flbl_to_hlbl : Dict
        Dictionary containing key:value pairs where keys are original classes seen during fit and 
        values are paths in the random generated tree.
    hlbl_to_flbl : Dict
        Reverse dictionary of flbl_to_hlbl.
    
    Examples
    --------
    >>> import utils
    >>> import numpy as np
    >>> y = np.random.choice(["A", "B", "C", "D", "E", "F", "G", 
    "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", 
    "R", "S", "T", "U", "V", "W", "X", "Y", "Z"],1000)
    >>> hlt = utils.FHLabelTransformer((2,4),sep=";",random_state=2021)
    >>> y_transform = hlt.fit_transform(y)
    >>> y_backtransform = hle.inverse_transform(y_transform)
    """
    def __init__(self, k=(2,2), sep=';', random_state=None):
        self.k = k
        self.sep = sep
        self.random_state = random_state

    def fit(self, y):
        """ Fit hierarchical label encoder.
        
        Parameters
        ----------
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        self : returns an instance of self.
        """
        self.random_state_ = check_random_state(self.random_state)
        y = column_or_1d(y, warn=True)
        # store classes seen during fit
        self.classes_ = list(np.unique(y))
        # label->path in random hierarchy dict
        self.flbl_to_hlbl = {c:[] for c in self.classes_}
        # now process each unique label and get path in random hierarchy
        lbls_to_process = [[c] for c in self.classes_]
        while len(lbls_to_process) > 1:
            self.random_state_.shuffle(lbls_to_process)
            ch_list = []
            for i in range(min(self.random_state_.randint(self.k[0], self.k[1]+1),len(lbls_to_process))):
                ch = lbls_to_process.pop(0)
                for c in ch:
                    self.flbl_to_hlbl[c].append(str(i))
                ch_list.extend(ch)
            lbls_to_process.append(ch_list)
        self.flbl_to_hlbl = {k: '.'.join((v+['r'])[::-1]) for k,v in self.flbl_to_hlbl.items()}
        # also store decoding dict
        self.hlbl_to_flbl = {v:k for k,v in self.flbl_to_hlbl.items()}

        return self

    def fit_transform(self, y):
        """ Fit hierarchical label encoder and return encoded labels.

        Parameters
        ----------
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        y_transformed : array-like of shape (n_samples,)
        """
        self.fit(y)
        y_transformed = self.transform(y)

        return y_transformed

    def transform(self, y):
        """ Transform flat labels to hierarchical encodings.

        Parameters
        ----------
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        y_transformed : array-like of shape (n_samples,)
        """
        check_is_fitted(self)
        y = column_or_1d(y, warn=True)
        y_transformed = []
        if len(y) == 0:
            # transform of empty array is empty array
            y_transformed = np.array([])
        else:
            for yi in y:
                path = self.flbl_to_hlbl[yi].split('.')
                y_transformed.append(self.sep.join(['.'.join(path[:i]) for i in range(1,len(path)+1)]))

        return y_transformed

    def inverse_transform(self, y):
        """Transform hierarchical labels back to original encodings.

        Parameters
        ----------
        y : ndarray of shape (n_samples,)
            Target values.

        Returns
        -------
        y_transformed : ndarray of shape (n_samples,)
        """
        check_is_fitted(self)
        y = column_or_1d(y, warn=True)
        y_transformed = []
        if len(y) == 0:
            # transform of empty array is empty array
            y_transformed = np.array([])
        else:
            for yi in y:
                path = yi.split(self.sep)[-1]
                y_transformed.append(self.hlbl_to_flbl[path])

        return y_transformed

class HFLabelTransformer(TransformerMixin, BaseEstimator):
    """ Hierarchical to flat label transformer, where flat labels are encoded as values between 0 and n_classes-1.

    Parameters
    ----------
    sep : str
        String used for path encodings.
    
    Attributes
    ----------
    sep : str
        String used for path encodings.
    classes_ : list 
        Classes (original) seen during fit.
    tree_ : Dict
        Dictionary which represents the hierarchy after fitting.
    hlbl_to_yhat_ : Dict
        Dictionary containing key:value pairs where keys are original classes seen during fit and 
        values are corresponding sets of encoded labels.
    yhat_to_hlbl_ : Dict
        Reverse dictionary of hlbl_to_yhat_
    hstruct_ : list
        List BFS structure which represents the hierarchy in terms of encoded labels after fit.
     
    Examples
    --------
    >>> y_h = np.array(["root;famA;genA","root;famA;genB","root;famB;genC","root;famB;genD"])
    >>> hle = utils.HFLabelTransformer(sep=";")
    >>> y_h_e = hle.fit_transform(y_h)
    >>> y_h_e_backtransform = hle.inverse_transform(y_h_e)
    """
    def __init__(self, sep=";"):
        self.sep = sep
    
    def fit(self, y):
        """ Fit hierarchical to flat label encoder.
        
        Parameters
        ----------
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        self : returns an instance of self.
        """
        y = column_or_1d(y, warn=True)
        # store classes seen during fit
        self.classes_ = list(np.unique(y))
        # now process labels and construct tree
        self.tree_ = {}
        for i,yi in enumerate(self.classes_):
            path_nodes = yi.split(self.sep)
            for j in range(1,len(path_nodes)+1):
                node = self.sep.join(path_nodes[:j])
                if node not in self.tree_:
                    self.tree_[node] = {
                        "yhat": [i],
                        "chn": [],
                        "par": (None if j==1 else self.sep.join(path_nodes[:j-1]))
                    }
                    if j!=1:
                        self.tree_[self.sep.join(path_nodes[:j-1])]["chn"].append(node)
                else:
                    if i not in self.tree_[node]["yhat"]:
                        self.tree_[node]["yhat"].append(i)
        # create hlbl->hpath dictionary
        self.hlbl_to_hpath_ = {c:[] for c in self.classes_}
        for c in self.classes_:
            node = c
            par_node = self.sep.join(c.split(self.sep)[:-1])
            while par_node is not None:
                self.hlbl_to_hpath_[c].append(self.tree_[par_node]["chn"].index(node))
                node = par_node
                par_node = self.tree_[node]["par"]
        # reverse values in hlbl->hpath
        self.hlbl_to_hpath_ = {k:v[::-1] for k,v in self.hlbl_to_hpath_.items()}
        # also store decoding dict
        self.hpath_to_hlbl_ = {str(v):k for k,v in self.hlbl_to_hpath_.items()}
        self.hlbl_to_yhat_ = {k:self.tree_[k]["yhat"] for k in self.tree_}
        self.yhat_to_hlbl_ = {str(v):k for k,v in self.hlbl_to_yhat_.items()}
        # and obtain struct (in terms of yhat)
        self.hstruct_ = []
        # find the root first 
        root = None
        for n in self.tree_:
            if self.tree_[n]["par"] is None:
                root = n
                break
        visit_list = [root]
        # now start constructing the struct
        while len(visit_list) != 0:
            node = visit_list.pop(0)
            self.hstruct_.append(self.hlbl_to_yhat_[node])
            visit_list.extend([nch for nch in self.tree_[node]["chn"]])

        return self

    def fit_transform(self, y, path=False):
        """ Fit hierarchical to flat label encoder and transform hierarchical labels to encoded flat labels.

        Parameters
        ----------
        y : array-like of shape (n_samples,)
            Target values.
        path : boolean, default=False
            Whether paths need to be returned or encoded flat labels.

        Returns
        -------
        y_transformed : array-like of shape (n_samples,)
        """
        self.fit(y)
        y_transformed = self.transform(y, path)

        return y_transformed

    def transform(self, y, path=False):
        """ Transform hierarchical labels to encoded flat labels or paths in hierarchy.

        Parameters
        ----------
        y : array-like of shape (n_samples,)
            Target values.
        path : boolean, default=False
            Whether paths need to be returned or encoded flat labels.

        Returns
        -------
        y_transformed : array-like of shape (n_samples,)
        """
        check_is_fitted(self)
        y = column_or_1d(y, warn=True)
        y_transformed = []
        if len(y) == 0:
            # transform of empty array is empty array
            y_transformed = np.array([])
        else:
            for yi in y:
                if not path:
                    y_transformed.append(self.hlbl_to_yhat_[yi])
                else:
                    y_transformed.append(self.hlbl_to_hpath_[yi])

        return y_transformed

    def inverse_transform(self, y, path=False):
        """ Transform encoded flat labels back to original hierarchical labels.

        Parameters
        ----------
        y : ndarray of shape (n_samples,)
            Target values.
        path : boolean, default=False
            Whether y represents paths or encoded flat labels.

        Returns
        -------
        y_transformed : ndarray of shape (n_samples,)
        """
        check_is_fitted(self)
        y_transformed = []
        if len(y) == 0:
            # transform of empty array is empty array
            y_transformed = np.array([])
        else:
            for yi in y:
                if not path:
                    y_transformed.append(self.yhat_to_hlbl_[str(yi)])
                else:
                    y_transformed.append(self.hpath_to_hlbl_[str(yi)])

        return y_transformed