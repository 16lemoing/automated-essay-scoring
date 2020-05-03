import random
import numpy as np
from sklearn.metrics import cohen_kappa_score
import pandas as pd
import matplotlib.pyplot as plt
from models import save_model_weights
from data import save_vocab

class Logger():
    
    def __init__(self, log_dir, checkpoint_dir, name, args, save_best_weights = False):
        
        # Build directories
        self.log_dir = log_dir
        self.checkpoint_dir = checkpoint_dir
        self.plot_dir = self.log_dir / "plots"
        if not self.log_dir.exists():
            self.log_dir.mkdir()
        if not self.plot_dir.exists():
            self.plot_dir.mkdir()
        if not self.checkpoint_dir.exists():
            self.checkpoint_dir.mkdir()
        self.id = self.generate_id()
        self.id_plot_dir = self.plot_dir / self.id
        if not self.id_plot_dir.exists():
            self.id_plot_dir.mkdir()
        self.save_best_weights = save_best_weights
        self.name = name
        self.args = vars(args)
        self.fold = -1
        self.train_losses = {}
        self.valid_losses = {}
        self.valid_weighted_kappas = {}
        self.valid_global_kappas = {}
        self.valid_individual_kappas = {}
    
    def generate_id(self):
        id = f"{random.randint(0, 99999):05d}"
        if (self.plot_dir / id).exists():
            return self.generate_id()
        else:
            return id
    
    def init_fold(self):
        self.fold += 1
        self.train_losses[self.fold] = []
        self.valid_losses[self.fold] = []
        self.valid_weighted_kappas[self.fold] = []
        self.valid_global_kappas[self.fold] = []
        self.valid_individual_kappas[self.fold] = []
    
    def log(self, train_loss = None, valid_loss = None, valid_weighted_kappa = None,
            valid_global_kappa = None, valid_individual_kappa = None):
        if train_loss is not None:
            self.train_losses[self.fold].append(train_loss)
        if valid_loss is not None:
            self.valid_losses[self.fold].append(valid_loss)
        if valid_weighted_kappa is not None:
            self.valid_weighted_kappas[self.fold].append(valid_weighted_kappa)
        if valid_global_kappa is not None:
            self.valid_global_kappas[self.fold].append(valid_global_kappa)
        if valid_individual_kappa is not None:
            self.valid_individual_kappas[self.fold].append(valid_individual_kappa)
    
    def save_plots(self):
        # loss plots
        for fold in self.train_losses:
            labels = ['train_loss']
            losses = [self.train_losses[fold]]
            if fold in self.valid_losses:
                losses.append(self.valid_losses[fold])
                labels.append('valid_loss')
            self.save_plot(f"fold{fold}_loss", "epoch", "loss", x = range(len(losses[0])), y = losses, labels = labels)
        # kappa plots
        for fold in self.valid_weighted_kappas:
            weighted_kappas = [self.valid_weighted_kappas[fold]]
            self.save_plot(f"fold{fold}_weighted_kappas", "epoch", "weighted_kappas", x = range(len(weighted_kappas[0])), y = weighted_kappas, labels = ['valid_weighted_kappas'])
        for fold in self.valid_global_kappas:
            global_kappas = [self.valid_global_kappas[fold]]
            self.save_plot(f"fold{fold}_global_kappas", "epoch", "global_kappas", x = range(len(global_kappas[0])), y = global_kappas, labels = ['valid_global_kappas'])
        for fold in self.valid_individual_kappas:
            individual_kappas = [[self.valid_individual_kappas[fold][i][set_idx] for i in range(len(self.valid_individual_kappas[fold]))] for set_idx in self.valid_individual_kappas[fold][0]]
            labels = [f"set_{set_idx}" for set_idx in self.valid_individual_kappas[fold][0]]
            self.save_plot(f"fold{fold}_individual_kappas", "epoch", "individual_kappas", x = range(len(individual_kappas[0])), y = individual_kappas, labels = labels)
    
    def save_plot(self, title, x_name, y_name, x, y, labels):
        fig, ax = plt.subplots()
        for i in range(len(y)):
            plt.plot(x, y[i], label = labels[i])
        plt.xlabel(x_name)
        plt.ylabel(y_name)
        plt.title(title)
        plt.legend(bbox_to_anchor = (1.05, 1.0), loc = 'upper left')
        plt.tight_layout()
        plt.savefig(self.id_plot_dir / f"{title}.pdf", format = 'pdf')
    
    def update_csv(self):
        csv_file = self.log_dir / f"{self.name}.csv"
        excel_file = self.log_dir / f"{self.name}.xlsx"
        if csv_file.exists():
            df = pd.read_csv(csv_file, sep = ',')
        else:
            columns = ["id"]
            columns += list(self.args.keys())
            columns += ["train_loss", "valid_loss", "valid_weigthed_kappa", "valid_global_kappa"]
            columns += [f"valid_set_{i}_kappa" for i in [1, 2, 3, 4, 5, 6, 7, 8]]
            df = pd.DataFrame(columns = columns)
        new_row = {key: self.args[key] for key in self.args}
        new_row["id"] = int(self.id)
        
        overall_metrics = self.get_overall_metrics()
        new_row["train_loss"] = overall_metrics[0]
        new_row["valid_loss"] = overall_metrics[1]
        new_row["valid_weigthed_kappa"] = overall_metrics[2]
        new_row["valid_global_kappa"] = overall_metrics[3]
        for i in [1, 2, 3, 4, 5, 6, 7, 8]:
            new_row[f"valid_set_{i}_kappa"] = overall_metrics[4][i]
        
        df = df.append(new_row, ignore_index = True)
        df.to_csv(csv_file, sep = ",", index = False)
        df.to_excel(excel_file)

    def get_overall_metrics(self):
        folds = list(self.train_losses.keys())
        try:
            best_epoch = {fold: np.argmin(self.valid_losses[fold]) for fold in folds}
        except:
            best_epoch = {fold: -1 for fold in folds}
        try:
            train_loss = np.mean([self.train_losses[fold][best_epoch[fold]] for fold in folds])
        except:
            train_loss = 0
        try:
            valid_loss = np.mean([self.valid_losses[fold][best_epoch[fold]] for fold in folds])
        except:
            valid_loss = 0
        try:
            valid_weighted_kappa = np.mean([self.valid_weighted_kappas[fold][best_epoch[fold]] for fold in folds])
        except:
            valid_weighted_kappa = 0
        try:
            valid_global_kappa = np.mean([self.valid_global_kappas[fold][best_epoch[fold]] for fold in folds])
        except:
            valid_global_kappa = 0
        valid_individual_kappa = {}
        for i in [1, 2, 3, 4, 5, 6, 7, 8]:
            try:
                kappa = np.mean([self.valid_individual_kappas[fold][best_epoch[fold]][i] for fold in folds])
            except:
                kappa = 0
            valid_individual_kappa[i] = kappa
        return train_loss, valid_loss, valid_weighted_kappa, valid_global_kappa, valid_individual_kappa
    
    def checkpoint_weights(self, model):
        if self.save_best_weights:
            out_dir = self.checkpoint_dir / self.id
            if not out_dir.exists():
                out_dir.mkdir()
            weights_file = out_dir / f"fold{self.fold}_weights.pth"
            save_model_weights(model, weights_file)
    
    def get_checkpoint_folder(self):
        return self.checkpoint_dir / self.id
    
    def checkpoint_vocab(self, vocab):
        out_dir = self.checkpoint_dir / self.id
        if not out_dir.exists():
            out_dir.mkdir()
        save_vocab(vocab, out_dir / 'vocab.pkl')

def get_kappa(scores, pred_scores, sets):
    scores = np.array(scores)
    pred_scores = np.array(pred_scores)
    sets = np.array(sets)
    set_idxs = list(set(sets))
    i_kappa = {}
    w_kappa = 0
    for idx in set_idxs:
        set_mask = (sets == idx)
        set_scores = scores[set_mask]
        set_pred_scores = pred_scores[set_mask]
        set_kappa = cohen_kappa_score(set_scores, set_pred_scores, weights = 'quadratic')
        i_kappa[idx] = set_kappa
        set_length = np.sum(set_mask)
        w_kappa += set_length * set_kappa
    w_kappa /= len(scores)
    g_kappa = cohen_kappa_score(scores, pred_scores, weights = 'quadratic')
    return w_kappa, g_kappa, i_kappa