from base.base_trainer import BaseTrainer
from base.base_dataset import BaseADDataset
from base.base_net import BaseNet
from torch.utils.data.dataloader import DataLoader
from sklearn.metrics import roc_auc_score, roc_curve, precision_score, recall_score, f1_score, confusion_matrix, precision_recall_curve

import logging
import time
import torch
import torch.optim as optim
import numpy as np


class DeepSVDDTrainer(BaseTrainer):

    def __init__(self, objective, R, c, nu: float, optimizer_name: str = 'adam', lr: float = 0.001, n_epochs: int = 150,
                 lr_milestones: tuple = (), batch_size: int = 128, weight_decay: float = 1e-6, device: str = 'cuda',
                 n_jobs_dataloader: int = 0):
        super().__init__(optimizer_name, lr, n_epochs, lr_milestones, batch_size, weight_decay, device,
                         n_jobs_dataloader)

        assert objective in ('one-class', 'soft-boundary'), "Objective must be either 'one-class' or 'soft-boundary'."
        self.objective = objective

        # Deep SVDD parameters
        self.R = torch.tensor(R, device=self.device)  # radius R initialized with 0 by default.
        self.c = torch.tensor(c, device=self.device) if c is not None else None
        self.nu = nu

        # Optimization parameters
        self.warm_up_n_epochs = 10  # number of training epochs for soft-boundary Deep SVDD before radius R gets updated

        # Results
        self.train_time = None
        self.test_auc = None
        self.test_time = None
        self.test_scores = None

    def train(self, dataset: BaseADDataset, net: BaseNet):
        logger = logging.getLogger()

        # Set device for network
        net = net.to(self.device)

        # Get train data loader
        train_loader, _ = dataset.loaders(batch_size=self.batch_size, num_workers=self.n_jobs_dataloader)

        # Set optimizer (Adam optimizer for now)
        optimizer = optim.Adam(net.parameters(), lr=self.lr, weight_decay=self.weight_decay,
                               amsgrad=self.optimizer_name == 'amsgrad')

        # Set learning rate scheduler
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.lr_milestones, gamma=0.1)

        # Initialize hypersphere center c (if c not loaded)
        if self.c is None:
            logger.info('Initializing center c...')
            self.c = self.init_center_c(train_loader, net)
            logger.info('Center c initialized.')

        # Training
        logger.info('Starting training...')
        start_time = time.time()
        net.train()
        for epoch in range(self.n_epochs):

            # Move scheduler.step() after optimizer.step()
            if epoch in self.lr_milestones:
                logger.info('  LR scheduler: new learning rate is %g' % float(scheduler.get_lr()[0]))

            loss_epoch = 0.0
            n_batches = 0
            epoch_start_time = time.time()
            for data in train_loader:
                inputs, _, _ = data
                inputs = inputs.to(self.device)

                # Zero the network parameter gradients
                optimizer.zero_grad()

                # Update network parameters via backpropagation: forward + backward + optimize
                outputs = net(inputs)
                dist = torch.sum((outputs - self.c) ** 2, dim=1)
                if self.objective == 'soft-boundary':
                    scores = dist - self.R ** 2
                    loss = self.R ** 2 + (1 / self.nu) * torch.mean(torch.max(torch.zeros_like(scores), scores))
                else:
                    loss = torch.mean(dist)
                loss.backward()
                optimizer.step()  # Call optimizer.step() first

                # Update hypersphere radius R on mini-batch distances
                if (self.objective == 'soft-boundary') and (epoch >= self.warm_up_n_epochs):
                    self.R.data = torch.tensor(get_radius(dist, self.nu), device=self.device)

                loss_epoch += loss.item()
                n_batches += 1

            scheduler.step()  # Then call scheduler.step()

            # log epoch statistics
            epoch_train_time = time.time() - epoch_start_time
            logger.info('  Epoch {}/{}\t Time: {:.3f}\t Loss: {:.8f}'
                        .format(epoch + 1, self.n_epochs, epoch_train_time, loss_epoch / n_batches))

        self.train_time = time.time() - start_time
        logger.info('Training time: %.3f' % self.train_time)

        logger.info('Finished training.')

        return net

    def test(self, dataset: BaseADDataset, net: BaseNet, threshold=None):
        logger = logging.getLogger()

        # Set device for network
        net = net.to(self.device)

        # Get test data loader
        _, test_loader = dataset.loaders(batch_size=self.batch_size, num_workers=self.n_jobs_dataloader)

        # Testing
        logger.info('Starting testing...')
        start_time = time.time()
        idx_label_score = []
        net.eval()
        with torch.no_grad():
            for data in test_loader:
                inputs, labels, idx = data
                inputs = inputs.to(self.device)
                outputs = net(inputs)
                dist = torch.sum((outputs - self.c) ** 2, dim=1)
                if self.objective == 'soft-boundary':
                    scores = dist - self.R ** 2
                else:
                    scores = dist

                # Save triples of (idx, label, score) in a list
                idx_label_score += list(zip(idx.cpu().data.numpy().tolist(),
                                            labels.cpu().data.numpy().tolist(),
                                            scores.cpu().data.numpy().tolist()))

        self.test_time = time.time() - start_time
        logger.info('Testing time: %.3f' % self.test_time)

        self.test_scores = idx_label_score

        # Compute AUC
        _, labels, scores = zip(*idx_label_score)
        labels = np.array(labels)
        scores = np.array(scores)

        self.test_auc = roc_auc_score(labels, scores)
        logger.info('Test set AUC: {:.2f}%'.format(100. * self.test_auc))

        # Compute Accuracy with calculated thresholds or user-provided threshold
        if threshold is None:
            # Using Youden's J statistic and F1 score
            fpr, tpr, thresholds = roc_curve(labels, scores)
            youden_j = tpr - fpr  # Youden's J statistic (maximizing sensitivity + specificity - 1)
            optimal_idx = np.argmax(youden_j)
            optimal_threshold_youden_j = thresholds[optimal_idx]
            # Using F1 score
            _, _, thresholds = precision_recall_curve(labels, scores)
            f1_scores = [f1_score(labels, (scores >= t).astype(int)) for t in thresholds]
            optimal_idx = np.argmax(f1_scores)
            optimal_threshold_f1 = thresholds[optimal_idx]
            
            logger.info('Calculated optimal thresholds: Youden J = {:.2f}, F1 = {:.2f}'.format(
                optimal_threshold_youden_j, optimal_threshold_f1))
        else:
            logger.info('Using user-provided threshold: {:.2f}'.format(threshold))
            optimal_threshold_youden_j = threshold
            optimal_threshold_f1 = threshold

        # Predict labels using the threshold(s)
        predictions_youden_j = (scores >= optimal_threshold_youden_j).astype(int)
        predictions_f1 = (scores >= optimal_threshold_f1).astype(int)

        accuracy_youden_j = np.mean(predictions_youden_j == labels)
        accuracy_f1 = np.mean(predictions_f1 == labels)

        if np.isclose(accuracy_youden_j, accuracy_f1):
            self.test_accuracy = accuracy_youden_j
            logger.info('Test set Accuracy: {:.2f}% (Same for both thresholds: {:.2f})'.format(100. * self.test_accuracy, optimal_threshold_youden_j))
            
            # Compute metrics based on Youden's J threshold
            self.test_conf_matrix = confusion_matrix(labels, predictions_youden_j)
            self.test_precision = precision_score(labels, predictions_youden_j)
            self.test_recall = recall_score(labels, predictions_youden_j)
            self.test_f1 = f1_score(labels, predictions_youden_j)
        else:
            self.test_accuracy = [accuracy_youden_j, accuracy_f1]
            logger.info('Test set Accuracy: {:.2f}% (Youden J Threshold: {:.2f})'.format(100. * self.test_accuracy[0], optimal_threshold_youden_j))
            logger.info('Test set Accuracy: {:.2f}% (F1 Threshold: {:.2f})'.format(100. * self.test_accuracy[1], optimal_threshold_f1))
            
            # Compute metrics based on both thresholds
            self.test_conf_matrix = {
            'youden_j': confusion_matrix(labels, predictions_youden_j),
            'f1': confusion_matrix(labels, predictions_f1)
            }
            self.test_precision = {
            'youden_j': precision_score(labels, predictions_youden_j),
            'f1': precision_score(labels, predictions_f1)
            }
            self.test_recall = {
            'youden_j': recall_score(labels, predictions_youden_j),
            'f1': recall_score(labels, predictions_f1)
            }
            self.test_f1 = {
            'youden_j': f1_score(labels, predictions_youden_j),
            'f1': f1_score(labels, predictions_f1)
            }
            
        logger.info('Test set Precision: {:.2f}'.format(self.test_precision if isinstance(self.test_precision, float) else self.test_precision['youden_j']))
        logger.info('Test set Recall: {:.2f}'.format(self.test_recall if isinstance(self.test_recall, float) else self.test_recall['youden_j']))
        logger.info('Test set F1 Score: {:.2f}'.format(self.test_f1 if isinstance(self.test_f1, float) else self.test_f1['youden_j']))
        logger.info('Test set Confusion Matrix:\n{}'.format(self.test_conf_matrix if isinstance(self.test_conf_matrix, np.ndarray) else self.test_conf_matrix['youden_j']))
        
        logger.info('Finished testing.')

    def init_center_c(self, train_loader: DataLoader, net: BaseNet, eps=0.1):
        """Initialize hypersphere center c as the mean from an initial forward pass on the data."""
        n_samples = 0
        c = torch.zeros(net.rep_dim, device=self.device)

        net.eval()
        with torch.no_grad():
            for data in train_loader:
                # get the inputs of the batch
                inputs, _, _ = data
                inputs = inputs.to(self.device)
                outputs = net(inputs)
                n_samples += outputs.shape[0]
                c += torch.sum(outputs, dim=0)

        c /= n_samples

        # If c_i is too close to 0, set to +-eps. Reason: a zero unit can be trivially matched with zero weights.
        c[(abs(c) < eps) & (c < 0)] = -eps
        c[(abs(c) < eps) & (c > 0)] = eps

        return c


def get_radius(dist: torch.Tensor, nu: float):
    """Optimally solve for radius R via the (1-nu)-quantile of distances."""
    return np.quantile(np.sqrt(dist.clone().data.cpu().numpy()), 1 - nu)
