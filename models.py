import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from enum import Enum
from typing import List, Tuple

import numpy as np
from sklearn.base import clone
from joblib import dump
from utils import format_clickable_path
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import BaseEstimator
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.dummy import DummyClassifier
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import HistGradientBoostingClassifier

# ---------------------------
# Hyperparameter Interfaces and Classes
# ---------------------------

class ModelParams(ABC):
    @abstractmethod
    def to_dict(self) -> dict:
        pass

    @abstractmethod
    def to_filename_suffix(self) -> str:
        pass

@dataclass(frozen=True)
class DummyParams(ModelParams):
    strategy: str = "most_frequent"  # Options: "most_frequent", "stratified"

    def to_dict(self):
        return asdict(self)

    def to_filename_suffix(self):
        return f"Dummy_{self.strategy}"

    @staticmethod
    def grid() -> List["DummyParams"]:
        return [DummyParams(strategy=s) for s in ["most_frequent", "stratified"]]

@dataclass(frozen=True)
class LogisticRegressionParams(ModelParams):
    solver: str = "liblinear"
    C: float = 1.0
    max_iter: int = 500
    class_weight: str = "balanced"
    verbose: int = 0

    def to_dict(self):
        return asdict(self)

    def to_filename_suffix(self):
        return f"LR_C{self.C}_iter{self.max_iter}"

    @staticmethod
    def grid() -> List["LogisticRegressionParams"]:
        return [
            LogisticRegressionParams(C=c, max_iter=it)
            for c in [0.1, 1.0, 10.0]
            for it in [100, 500]
        ]

@dataclass(frozen=True)
class TorchLogisticRegressionParams(ModelParams):
    learning_rate: float = 0.001
    batch_size: int = 512
    max_epochs: int = 200

    def to_dict(self): return asdict(self)

    def to_filename_suffix(self):
        return f"TorchLR_lr{self.learning_rate}_e{self.max_epochs}_bs{self.batch_size}"

    @staticmethod
    def grid():
        return [TorchLogisticRegressionParams(learning_rate=lr, batch_size=batch_size) for lr in [0.001, 0.01, 0.1] for batch_size in [16, 48, 512]]


@dataclass(frozen=True)
class RandomForestParams(ModelParams):
    n_estimators: int = 50
    max_depth: int = 10  # Set a reasonable default
    class_weight: str = "balanced"
    random_state: int = 42
    verbose: int = 0
    n_jobs: int = -1

    def to_dict(self):
        return asdict(self)

    def to_filename_suffix(self):
        return f"RF_ne{self.n_estimators}_md{self.max_depth}"

    @staticmethod
    def grid() -> List["RandomForestParams"]:
        return [
            RandomForestParams(n_estimators=n, max_depth=md)
            for n in [10, 20, 50]
            for md in [5, 10, 15]
        ]
        
@dataclass(frozen=True)
class HistGradientBoostingParams(ModelParams):
    max_iter: int = 100
    learning_rate: float = 0.1
    max_depth: int = 10
    early_stopping: bool = True
    random_state: int = 42
    verbose: int = 0

    def to_dict(self):
        return asdict(self)

    def to_filename_suffix(self):
        return f"HGBT_iter{self.max_iter}_lr{self.learning_rate}_md{self.max_depth}"

    @staticmethod
    def grid() -> List["HistGradientBoostingParams"]:
        return [
            HistGradientBoostingParams(max_iter=iter_, learning_rate=lr, max_depth=md)
            for iter_ in [50, 100]
            for lr in [0.05, 0.1]
            for md in [5, 10]
        ]
        
@dataclass(frozen=True)
class SGDParams(ModelParams):
    loss: str = "log_loss"  # Logistic Regression loss
    penalty: str = "l2"
    alpha: float = 0.0001
    max_iter: int = 1000
    random_state: int = 42

    def to_dict(self):
        return asdict(self)

    def to_filename_suffix(self):
        return f"SGD_loss{self.loss}_penalty{self.penalty}_alpha{self.alpha}_iter{self.max_iter}"

    @staticmethod
    def grid() -> List["SGDParams"]:
        return [
            SGDParams(loss=l, penalty=p, alpha=a, max_iter=it)
            for l in ["log_loss", "hinge"]  # Logistic Regression and Linear SVM
            for p in ["l2", "l1"]
            for a in [0.0001, 0.001]
            for it in [500, 1000]
        ]
        
@dataclass(frozen=True)
class MLPParams(ModelParams):
    hidden_layer_sizes: Tuple[int, ...] = (100,)
    alpha: float = 0.0001
    max_iter: int = 300

    def to_dict(self):
        return asdict(self)

    def to_filename_suffix(self):
        return f"MLP_hls{'x'.join(map(str, self.hidden_layer_sizes))}_alpha{self.alpha}"

    @staticmethod
    def grid() -> List["MLPParams"]:
        return [
            MLPParams(hidden_layer_sizes=hls, alpha=a)
            for hls in [(100,), (100, 50)]
            for a in [0.0001, 0.001]
        ]

class ManualMultiOutput:
    def __init__(self, classifiers):
        self.classifiers = classifiers

    def predict(self, X):
        results = []
        for clf in self.classifiers:
            if clf is None:
                results.append(np.zeros(X.shape[0], dtype=int))
            else:
                results.append(clf.predict(X))
        return np.column_stack(results)

    def predict_proba(self, X):
        return [
            np.full((X.shape[0], 2), [1.0, 0.0]) if clf is None else clf.predict_proba(X)
            for clf in self.classifiers
        ]

# ---------------------------
# Model Classes
# ---------------------------

class ModelResult:
    def __init__(self, trained_model, model_config, params : ModelParams, feature_key: str, best_epoch: int = None):
        if isinstance(trained_model, ModelResult):
            raise TypeError("trained_model should not be an instance of ModelResult. Did you accidentally wrap it twice?")
        
        self.trained_model = trained_model  # Could be a list of classifiers or a Torch model
        self.model_config = model_config
        self.params : ModelParams = params
        self.feature_key = feature_key
        self.best_epoch = best_epoch

    def predict(self, X):
        return self.model_config.value.predict(self, X)

    def predict_proba(self, X):
        return self.model_config.value.predict_proba(self, X)

    def model_name(self):
        return f"{self.model_config.name}_{self.params.to_filename_suffix()}_{self.feature_key}"

    @staticmethod
    def derive_model_name(config, params, feature_key: str) -> str:
        return f"{config.name}_{params.to_filename_suffix()}_{feature_key}"

    def save(self, output_dir):
        filename = f"{self.model_name()}.joblib"
        model_path = os.path.join(output_dir, filename)
        dump(self, model_path)
        print(f"Model saved to {model_path}")

class TrainingMonitor(ABC):
    @abstractmethod
    def should_stop(self, epoch: int, model_result: ModelResult) -> bool:
        """Decides whether to stop training early based on the current model state."""
        pass
    
class EarlyStoppingMonitor(TrainingMonitor):
    def __init__(self, X_val, Y_val, patience=10, min_delta=0.0001, metric="roc_auc_micro"):
        self.X_val = X_val
        self.Y_val = Y_val
        if self.X_val.shape[0] != self.Y_val.shape[0]:
            raise ValueError(f"Shape mismatch: X_val has {self.X_val.shape[0]} samples, but Y_val has {self.Y_val.shape[0]} samples.")
        self.patience = patience
        self.min_delta = min_delta
        self.metric = metric
        self.best_score = None
        self.best_epoch = None
        self.counter = 0
        self.progress_callback = None

    def should_stop(self, epoch: int, model_result: ModelResult) -> bool:

        Y_pred_proba = model_result.predict_proba(self.X_val)
        
        if Y_pred_proba.shape != self.Y_val.shape:
            raise ValueError(f"Prediction shape mismatch: predicted {Y_pred_proba.shape}, expected {self.Y_val.shape}.")
        
        # Compute metric using standard functions
        try:
            if self.metric == "roc_auc_micro":                
                score = roc_auc_score(self.Y_val, np.asarray(Y_pred_proba), average="micro")
            else:
                raise ValueError(f"Unsupported metric: {self.metric}")
        except ValueError as e:
            print(str(e))
            return False

        if self.best_score is None or score > self.best_score + self.min_delta:
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1

        if self.progress_callback:
            self.progress_callback(epoch, score)

        return self.counter >= self.patience

class MultiLabelModel(ABC):
    @abstractmethod
    def create_base_estimator(self, params: ModelParams): pass

    @abstractmethod
    def fit(self, X_train, Y_train, params: ModelParams, feature_key: str, early_stopping_monitor : EarlyStoppingMonitor = None): pass

    @abstractmethod
    def predict(self, trained_model, X): pass

    @abstractmethod
    def hyperparameter_grid(self): pass
    
    def supports_epochs(self): 
        return False

class DummyModel(MultiLabelModel):
    def create_base_estimator(self, params: DummyParams):
        return DummyClassifier(**params.to_dict())

    def fit(self, X_train, Y_train, params: DummyParams, feature_key: str, early_stopping_monitor : EarlyStoppingMonitor = None) -> ModelResult:
        base_estimator = self.create_base_estimator(params)
        classifiers = []

        for i in tqdm(range(Y_train.shape[1]), desc="Training Dummy Classifiers"):
            y = Y_train[:, i]
            if np.unique(y).size < 2:
                classifiers.append(None)
                continue
            clf = clone(base_estimator)
            clf.fit(X_train, y)
            classifiers.append(clf)

        return ModelResult(classifiers, ModelConfig.DUMMY, params, feature_key)

    def predict(self, model_result: ModelResult, X):
        classifiers = model_result.trained_model  # This should be the list of classifiers

        return np.column_stack([
            np.zeros(X.shape[0], dtype=int) if clf is None else clf.predict(X)
            for clf in classifiers
        ])

    def predict_proba(self, model_result: ModelResult, X):
        classifiers = model_result.trained_model
        return np.column_stack([
            np.zeros(X.shape[0]) if clf is None else clf.predict_proba(X)[:, 1]
            for clf in classifiers
        ])
        
    def hyperparameter_grid(self):
        return DummyParams.grid()

class LogisticRegressionModel(MultiLabelModel):
    def create_base_estimator(self, params: LogisticRegressionParams):
        return LogisticRegression(**params.to_dict())

    def fit(self, X_train, Y_train, params, feature_key : str, early_stopping_monitor : EarlyStoppingMonitor = None):
        base_estimator = self.create_base_estimator(params)
        classifiers = []
        for i in tqdm(range(Y_train.shape[1]), desc="Training LR Classifiers"):
            y = Y_train[:, i]
            if np.unique(y).size < 2:
                classifiers.append(None)
                continue
            clf = clone(base_estimator)
            clf.fit(X_train, y)
            classifiers.append(clf)
        return classifiers

    def predict(self, model_result : ModelResult, X):
        trained_model = model_result.trained_model
        return np.column_stack([
            np.zeros(X.shape[0], dtype=int) if clf is None else clf.predict(X)
            for clf in trained_model
        ])

    def hyperparameter_grid(self):
        return LogisticRegressionParams.grid()

class RandomForestModel(MultiLabelModel):
    def create_base_estimator(self, params: RandomForestParams):
        return RandomForestClassifier(**params.to_dict())

    def fit(self, X_train, Y_train, params: RandomForestParams, feature_key: str, early_stopping_monitor : EarlyStoppingMonitor = None) -> ModelResult:
        base_estimator = self.create_base_estimator(params)
        classifiers = []

        for i in tqdm(range(Y_train.shape[1]), desc="Training RF Classifiers"):
            y = Y_train[:, i]
            if np.unique(y).size < 2:
                classifiers.append(None)
                continue
            clf = clone(base_estimator)
            clf.fit(X_train, y)
            classifiers.append(clf)

        return ModelResult(classifiers, ModelConfig.RANDOM_FOREST, params, feature_key)

    def predict(self, model_result: ModelResult, X):
        classifiers = model_result.trained_model

        return np.column_stack([
            np.zeros(X.shape[0], dtype=int) if clf is None else clf.predict(X)
            for clf in classifiers
        ])

    def predict_proba(self, model_result: ModelResult, X):
        classifiers = model_result.trained_model
        return np.column_stack([
            np.zeros(X.shape[0]) if clf is None else clf.predict_proba(X)[:, 1]
            for clf in classifiers
        ])

    def hyperparameter_grid(self):
        return RandomForestParams.grid()

class HistGradientBoostingModel(MultiLabelModel):
    def create_base_estimator(self, params: HistGradientBoostingParams):
        return HistGradientBoostingClassifier(**params.to_dict())

    def fit(self, X_train, Y_train, params: HistGradientBoostingParams, feature_key: str, early_stopping_monitor: EarlyStoppingMonitor = None) -> ModelResult:
        base_estimator = self.create_base_estimator(params)
        classifiers = []

        for i in tqdm(range(Y_train.shape[1]), desc="Training HGBT Classifiers"):
            y = Y_train[:, i]
            if np.unique(y).size < 2:
                classifiers.append(None)
                continue
            clf = clone(base_estimator)
            clf.fit(X_train, y)
            classifiers.append(clf)

        return ModelResult(classifiers, ModelConfig.HIST_GRADIENT_BOOSTING, params, feature_key)

    def predict(self, model_result: ModelResult, X):
        classifiers = model_result.trained_model
        return np.column_stack([
            np.zeros(X.shape[0], dtype=int) if clf is None else clf.predict(X)
            for clf in classifiers
        ])

    def predict_proba(self, model_result: ModelResult, X):
        classifiers = model_result.trained_model
        return np.column_stack([
            np.zeros(X.shape[0]) if clf is None else clf.predict_proba(X)[:, 1]
            for clf in classifiers
        ])

    def hyperparameter_grid(self):
        return HistGradientBoostingParams.grid()
    
class TorchLogisticRegressionModel(MultiLabelModel):
    class TorchLRNet(nn.Module):
        def __init__(self, input_dim, num_classes):
            super().__init__()
            self.linear = nn.Linear(input_dim, num_classes)

        def forward(self, x):
            return self.linear(x)

    def create_base_estimator(self, params):
        return None  # Not needed for PyTorch

    def fit(self, X_train, Y_train, params: TorchLogisticRegressionParams, feature_key : str, early_stopping_monitor : EarlyStoppingMonitor = None):
        device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
        
        print("Using device : "+str(device))
        
        if(early_stopping_monitor is None) :
            raise ValueError("Early stopping monitor must be passed here!")
            
        input_dim, num_classes = X_train.shape[1], Y_train.shape[1]
        model = self.TorchLRNet(input_dim, num_classes).to(device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=params.learning_rate)

        train_dataset = TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(Y_train, dtype=torch.float32)
        )
        train_loader = DataLoader(train_dataset, batch_size=params.batch_size, shuffle=True)
        
        progress_bar = tqdm(range(params.max_epochs), desc="Torch LR Training")

        def update_progress(epoch, score):
            progress_bar.set_description(f"Epoch {epoch}, ROC-AUC Micro: {score:.4f}")

        early_stopping_monitor.progress_callback = update_progress

        for epoch in progress_bar:
            model.train()
            for X_batch, Y_batch in train_loader:
                X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, Y_batch)
                loss.backward()
                optimizer.step()
                
            if early_stopping_monitor.should_stop(epoch+1, ModelResult(model, ModelConfig.TORCH_LOGISTIC_REGRESSION, params, feature_key)):
                print(f"Early stopping triggered at epoch {epoch + 1}.")
                break

        return ModelResult(model, ModelConfig.TORCH_LOGISTIC_REGRESSION, params, feature_key, early_stopping_monitor.best_epoch)
    
    def predict(self, model_result : ModelResult, X):
        trained_model = model_result.trained_model        
        device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
        trained_model.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
            logits = trained_model(X_tensor)
            preds = (torch.sigmoid(logits) > 0.5).int().cpu().numpy()
        return preds
    
    def predict_proba(self, model_result: ModelResult, X):
        trained_model = model_result.trained_model
        device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
        trained_model.eval()
        with torch.no_grad():
            inputs = torch.tensor(X, dtype=torch.float32).to(device)
            outputs = trained_model(inputs).sigmoid()
            return outputs.cpu().numpy()

    def hyperparameter_grid(self):
        return TorchLogisticRegressionParams.grid()

    def supports_epochs(self): 
        return True

from sklearn.linear_model import SGDClassifier

class SGDModel(MultiLabelModel):
    def create_base_estimator(self, params: SGDParams):
        return SGDClassifier(**params.to_dict())

    def fit(self, X_train, Y_train, params: SGDParams, feature_key: str, early_stopping_monitor: EarlyStoppingMonitor = None) -> ModelResult:
        base_estimator = self.create_base_estimator(params)
        classifiers = []

        for i in tqdm(range(Y_train.shape[1]), desc="Training SGD Classifiers"):
            y = Y_train[:, i]
            if np.unique(y).size < 2:
                classifiers.append(None)
                continue
            clf = clone(base_estimator)
            clf.fit(X_train, y)
            classifiers.append(clf)

        return ModelResult(classifiers, ModelConfig.SGD, params, feature_key)

    def predict(self, model_result: ModelResult, X):
        classifiers = model_result.trained_model
        return np.column_stack([
            np.zeros(X.shape[0], dtype=int) if clf is None else clf.predict(X)
            for clf in classifiers
        ])

    def predict_proba(self, model_result: ModelResult, X):
        classifiers = model_result.trained_model
        return np.column_stack([
            np.zeros(X.shape[0]) if clf is None else clf.predict_proba(X)[:, 1]
            for clf in classifiers
        ])

    def hyperparameter_grid(self):
        return SGDParams.grid()

# ---------------------------
# Model Config Enumeration
# ---------------------------

class ModelConfig(Enum):
    DUMMY = DummyModel()
    TORCH_LOGISTIC_REGRESSION = TorchLogisticRegressionModel()
    RANDOM_FOREST = RandomForestModel()
    HIST_GRADIENT_BOOSTING = HistGradientBoostingModel()
    #SVM = SVMModel() way too slow for our dataset, both in rbm and linear modes
    SGD = SGDModel()


#model fitting / training

def train_multi_label_model(X_train, Y_train, X_val, Y_val, model_config: ModelConfig, params: ModelParams, feature_key: str) -> ModelResult:
    model_class : MultiLabelModel = model_config.value
    early_stopping_monitor = EarlyStoppingMonitor(X_val, Y_val, 10)    
    trained_model : ModelResult = model_class.fit(X_train, Y_train, params, feature_key, early_stopping_monitor=early_stopping_monitor)
    return trained_model

def save_model(model_result : ModelResult, output_dir):
    filename = f"{model_result.model_name()}.joblib"
    model_path = os.path.join(output_dir, filename)
    dump(model_result, model_path)
    print(f"Model saved to {format_clickable_path(model_path)}")