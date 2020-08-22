import os
import json
# import pytorch and transormers
import torch
import transformers
# import visualization tools
from tqdm import tqdm
from matplotlib import pyplot as plt


class BaseDataset(torch.utils.data.TensorDataset):
    """ Base Class for Datasets used by Trainers """

    def __init__(self, train:bool, tokenizer:transformers.BertTokenizer, seq_length:int, data_base_dir:str):
        raise NotImplementedError()


class BaseTrainer(object):
    """ Base Class for Trainers """

    # define dataset base class the trainer can use
    BASE_DATASET_TYPE = BaseDataset

    def __init__(self, 
        # model and tokenizer
        bert_model_type:type,
        tokenizer_type:type,
        bert_base_model:str,
        device:str,
        # data
        dataset_type:torch.utils.data.Dataset,
        data_base_dir:str,
        seq_length:int,
        batch_size:int,
        # optimizer
        learning_rate:float,
        weight_decay:float,
    ):
        # save values
        self.device = device
        self.bert_base_model = bert_base_model
        self.lr, self.wd = learning_rate, weight_decay
        # create tokenizer
        if tokenizer_type is not None:
            self.tokenizer = tokenizer_type.from_pretrained(bert_base_model)
        # create model and optimizer
        if bert_model_type is not None:
            self.model = bert_model_type.from_pretrained(bert_base_model).to(device)
            self.optim = transformers.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.wd)
        # create dataloader
        if dataset_type is not None:
            assert tokenizer_type is not None
            # check dataset type
            if not issubclass(dataset_type, self.__class__.BASE_DATASET_TYPE):
                raise ValueError("Dataset Type %s must inherit %s!" % (dataset_type.__name__, self.__class__.BASE_DATASET_TYPE.__name__))
            # initialize dataloaders
            self.dataset_name = dataset_type.__name__
            self.train_dataloader = torch.utils.data.DataLoader(dataset_type(True, self.tokenizer, seq_length, data_base_dir), shuffle=True, batch_size=batch_size)
            self.test_dataloader = torch.utils.data.DataLoader(dataset_type(False, self.tokenizer, seq_length, data_base_dir), batch_size=batch_size)
        # save training metrics
        self.metrics = None


    def get_batch_loss(self, *batch) -> torch.tensor:
        raise NotImplementedError()

    def evaluate_batch(self, *batch) -> tuple:
        raise NotImplementedError()

    def compute_metrics(self, eval_caches:list) -> tuple:
        raise NotImplementedError()


    def run_epoch(self):

        # train model
        self.model.train()
        train_running_loss = 0
        # create progress bar
        with tqdm(total=len(self.train_dataloader)) as pbar:
            pbar.set_description("Train")

            for i, batch in enumerate(self.train_dataloader, 1):
                # get loss
                loss = self.compute_batch_loss(*batch)
                train_running_loss += loss.item()
                # backpropagate and update parameters
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

                # update progress bar
                pbar.set_postfix({'loss': train_running_loss / i})
                pbar.update(1)

        # test model
        self.model.eval()
        test_running_loss, eval_caches = 0, []
        # no gradients needed for evaluation
        with torch.no_grad():
            # create progress bar
            with tqdm(total=len(self.test_dataloader)) as pbar:
                pbar.set_description("Test")

                for i, batch in enumerate(self.train_dataloader, 1):
                    # evaluate batch
                    loss, cache = self.evaluate_batch(*batch)
                    test_running_loss += loss.item()
                    eval_caches.append(cache)
                    # update progress bar
                    pbar.set_postfix({'loss': test_running_loss / i})
                    pbar.update(1)

        # compute metrics and stuff
        metrics = self.compute_metrics(eval_caches)
        assert type(metrics) is tuple
        # return all metrics
        return (
            train_running_loss / len(self.train_dataloader), 
            test_running_loss / len(self.test_dataloader),
        ) + metrics

    def train(self, epochs:int) -> None:

        metric_caches = []
        # run epochs
        for e in range(1, epochs + 1):
            print("Epoch %i" % e)
            # run epoch
            metrics = self.run_epoch()
            metric_caches.append(metrics)
            # print
            print("Evaluation: %s" % ', '.join(["%.3f" % m for m in metrics]))
        # build metric lists
        self.metrics = tuple(zip(*metric_caches))

    def dump(self, dump_base_path:str):
        # create full path to dump directory
        dump_dir = os.path.join(
            dump_base_path, 
            self.model.__class__.__name__, 
            "%s-%s" % (self.bert_base_model, self.dataset_name)
        )
        # create directory
        os.makedirs(dump_dir, exist_ok=True)
        # save trainer setup in directory
        with open(os.path.join(dump_dir, "trainer.json"), 'w+') as f:
            f.write(json.dumps({
                'bert-base-model': self.bert_base_model,
                'dataset': self.dataset_name,
                'learning-rate': self.lr,
                'weight-decay': self.wd
            }, indent=4))
        # save plot
        self.plot().savefig(os.path.join(dump_dir, 'metrics.png'))
        plt.close()
        # save model and optimizer
        self.model.save_pretrained(dump_dir)
        torch.save(self.optim.state_dict(), os.path.join(dump_dir, 'optimizer.bin'))

    def plot(self, figsize=(8, 5)):
        # create figure
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        # plot train and test loss
        ax.plot(self.metrics[0], label='Train')
        ax.plot(self.metrics[1], label='Test')
        ax.legend()
        ax.set(xlabel='Epoch', ylabel='Loss')
        # return figure
        return fig