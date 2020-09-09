from __future__ import absolute_import, division, print_function

import json
import logging
import math
import os
import random
import warnings
from dataclasses import asdict
from multiprocessing import cpu_count

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score
from tensorboardX import SummaryWriter
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from tqdm.auto import tqdm, trange
from transformers import (
    WEIGHTS_NAME,
    AdamW,
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
    BertConfig,
    BertForTokenClassification,
    BertTokenizer,
    CamembertConfig,
    CamembertForTokenClassification,
    CamembertTokenizer,
    DistilBertConfig,
    DistilBertForTokenClassification,
    DistilBertTokenizer,
    ElectraConfig,
    ElectraForTokenClassification,
    ElectraTokenizer,
    LongformerConfig,
    LongformerForTokenClassification,
    LongformerTokenizer,
    MobileBertConfig,
    MobileBertForTokenClassification,
    MobileBertTokenizer,
    RobertaConfig,
    RobertaForTokenClassification,
    RobertaTokenizer,
    XLMRobertaConfig,
    XLMRobertaForTokenClassification,
    XLMRobertaTokenizer,
    get_linear_schedule_with_warmup,
)
from wandb import config

from simpletransformers.config.global_args import global_args
from simpletransformers.config.model_args import NERArgs
from simpletransformers.ner.ner_utils import (
    InputExample,
    LazyNERDataset,
    convert_examples_to_features,
    get_examples_from_df,
    get_labels,
    read_examples_from_file,
)

try:
    import wandb

    wandb_available = True
except ImportError:
    wandb_available = False

logger = logging.getLogger(__name__)


class NERModelDistillationMultiTask:
    def __init__( # Pass the trained model in model_teacher
        self, model_teacher, model_type, model_name, labels=None, args=None, use_cuda=True, cuda_device=-1, **kwargs,
    ):
        """
        Initializes a NERModel
        Args:
            model_type: The type of model (bert, roberta)
            model_name: Default Transformer model name or path to a directory containing Transformer model file (pytorch_nodel.bin).
            labels (optional): A list of all Named Entity labels.  If not given, ["O", "B-MISC", "I-MISC",  "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"] will be used.
            args (optional): Default args will be used if this parameter is not provided. If provided, it should be a dict containing the args that should be changed in the default args.
            use_cuda (optional): Use GPU if available. Setting to False will force model to use CPU only.
            cuda_device (optional): Specific GPU that should be used. Will use the first available GPU by default.
            **kwargs (optional): For providing proxies, force_download, resume_download, cache_dir and other options specific to the 'from_pretrained' implementation where this will be supplied.
        """  # noqa: ignore flake8"

        MODEL_CLASSES = {
            "auto": (AutoConfig, AutoTokenizer, AutoModelForTokenClassification),
            "bert": (BertConfig, BertForTokenClassification, BertTokenizer),
            "camembert": (CamembertConfig, CamembertForTokenClassification, CamembertTokenizer),
            "distilbert": (DistilBertConfig, DistilBertForTokenClassification, DistilBertTokenizer),
            "electra": (ElectraConfig, ElectraForTokenClassification, ElectraTokenizer),
            "longformer": (LongformerConfig, LongformerForTokenClassification, LongformerTokenizer),
            "roberta": (RobertaConfig, RobertaForTokenClassification, RobertaTokenizer),
            "xlmroberta": (XLMRobertaConfig, XLMRobertaForTokenClassification, XLMRobertaTokenizer),
        }
        self.model_teacher = model_teacher
        self.args = self._load_model_args(model_name)

        if isinstance(args, dict):
            self.args.update_from_dict(args)
        elif isinstance(args, NERArgs):
            self.args = args

        if "sweep_config" in kwargs:
            sweep_config = kwargs.pop("sweep_config")
            sweep_values = {key: value["value"] for key, value in sweep_config.as_dict().items() if key != "_wandb"}
            self.args.update_from_dict(sweep_values)

        if self.args.manual_seed:
            random.seed(self.args.manual_seed)
            np.random.seed(self.args.manual_seed)
            torch.manual_seed(self.args.manual_seed)
            if self.args.n_gpu > 0:
                torch.cuda.manual_seed_all(self.args.manual_seed)

        if not use_cuda:
            self.args.fp16 = False

        if labels and self.args.labels_list:
            assert labels == self.args.labels_list
            self.args.labels_list = labels
        elif labels:
            self.args.labels_list = labels
        elif self.args.labels_list:
            pass
        else:
            self.args.labels_list = [
                "O",
                "B-MISC",
                "I-MISC",
                "B-PER",
                "I-PER",
                "B-ORG",
                "I-ORG",
                "B-LOC",
                "I-LOC",
            ]
        self.num_labels = len(self.args.labels_list)

        config_class, model_class, tokenizer_class = MODEL_CLASSES[model_type]
        if self.num_labels:
            self.config = config_class.from_pretrained(model_name, num_labels=self.num_labels, **self.args.config)
            self.num_labels = self.num_labels
        else:
            self.config = config_class.from_pretrained(model_name, **self.args.config)
            self.num_labels = self.config.num_labels

        self.config.output_hidden_states=True
        if not self.args.quantized_model:
            self.model = model_class.from_pretrained(model_name, config=self.config, **kwargs)
        else:
            quantized_weights = torch.load(os.path.join(model_name, "pytorch_model.bin"))
            self.model = model_class.from_pretrained(None, config=self.config, state_dict=quantized_weights)

        if self.args.dynamic_quantize:
            self.model = torch.quantization.quantize_dynamic(self.model, {torch.nn.Linear}, dtype=torch.qint8)
        if self.args.quantized_model:
            self.model.load_state_dict(quantized_weights)
        if self.args.dynamic_quantize:
            self.args.quantized_model = True

        if use_cuda:
            if torch.cuda.is_available():
                if cuda_device == -1:
                    self.device = torch.device("cuda")
                else:
                    self.device = torch.device(f"cuda:{cuda_device}")
            else:
                raise ValueError(
                    "'use_cuda' set to True when cuda is unavailable."
                    "Make sure CUDA is available or set use_cuda=False."
                )
        else:
            self.device = "cpu"

        self.results = {}

        self.tokenizer = tokenizer_class.from_pretrained(model_name, do_lower_case=self.args.do_lower_case, **kwargs)

        self.args.model_name = model_name
        self.args.model_type = model_type

        self.pad_token_label_id = KLDivLoss().ignore_index

        if model_type == "camembert":
            warnings.warn(
                "use_multiprocessing automatically disabled as CamemBERT"
                " fails when using multiprocessing for feature conversion."
            )
            self.args.use_multiprocessing = False

        if self.args.wandb_project and not wandb_available:
            warnings.warn("wandb_project specified but wandb is not available. Wandb disabled.")
            self.args.wandb_project = None

    def train_model(
        self, train_data, output_dir=None, show_running_loss=True, args=None, eval_data=None, verbose=True, **kwargs
    ):
        """
        Trains the model using 'train_data'
        Args:
            train_data: train_data should be the path to a .txt file containing the training data OR a pandas DataFrame with 3 columns.
                        If a text file is given the data should be in the CoNLL format. i.e. One word per line, with sentences seperated by an empty line.
                        The first word of the line should be a word, and the last should be a Name Entity Tag.
                        If a DataFrame is given, each sentence should be split into words, with each word assigned a tag, and with all words from the same sentence given the same sentence_id.
            eval_data: Evaluation data (same format as train_data) against which evaluation will be performed when evaluate_during_training is enabled. Is required if evaluate_during_training is enabled.
            output_dir: The directory where model files will be saved. If not given, self.args.output_dir will be used.
            show_running_loss (optional): Set to False to prevent running loss from being printed to console. Defaults to True.
            args (optional): Optional changes to the args dict of the model. Any changes made will persist for the model.
            **kwargs: Additional metrics that should be used. Pass in the metrics as keyword arguments (name of metric: function to use). E.g. f1=sklearn.metrics.f1_score.
                        A metric function should take in two parameters. The first parameter will be the true labels, and the second parameter will be the predictions.
        Returns:
            None
        """  # noqa: ignore flake8"

        if args:
            self.args.update_from_dict(args)

        if self.args.silent:
            show_running_loss = False

        if self.args.evaluate_during_training and eval_data is None:
            if "eval_df" in kwargs:
                warnings.warn(
                    "The eval_df parameter has been renamed to eval_data."
                    " Using eval_df will raise an error in a future version."
                )
                eval_data = kwargs.pop("eval_df")
            else:
                raise ValueError(
                    "evaluate_during_training is enabled but eval_data is not specified."
                    " Pass eval_data to model.train_model() if using evaluate_during_training."
                )

        if not output_dir:
            output_dir = self.args.output_dir

        if os.path.exists(output_dir) and os.listdir(output_dir) and not self.args.overwrite_output_dir:
            raise ValueError(
                "Output directory ({}) already exists and is not empty."
                " Use --overwrite_output_dir to overcome.".format(output_dir)
            )

        self._move_model_to_device()

        train_dataset = self.load_and_cache_examples(train_data)

        os.makedirs(output_dir, exist_ok=True)

        global_step, tr_loss = self.train(
            train_dataset, output_dir, show_running_loss=show_running_loss, eval_data=eval_data, **kwargs
        )

        self.save_model(model=self.model)

        logger.info(" Training of {} model complete. Saved to {}.".format(self.args.model_type, output_dir))

    def train(self, train_dataset, output_dir, show_running_loss=True, eval_data=None, verbose=True, **kwargs):
        """
        Trains the model on train_dataset.
        Utility function to be used by the train_model() method. Not intended to be used directly.
        """

        device = self.device
        model = self.model
        model_teacher = self.model_teacher
        args = self.args

        tb_writer = SummaryWriter(logdir=args.tensorboard_dir)
        train_sampler = RandomSampler(train_dataset)
        train_dataloader = DataLoader(
            train_dataset,
            sampler=train_sampler,
            batch_size=args.train_batch_size,
            num_workers=self.args.dataloader_num_workers,
        )

        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

        no_decay = ["bias", "LayerNorm.weight"]

        optimizer_grouped_parameters = []
        custom_parameter_names = set()
        for group in self.args.custom_parameter_groups:
            params = group.pop("params")
            custom_parameter_names.update(params)
            param_group = {**group}
            param_group["params"] = [p for n, p in model.named_parameters() if n in params]
            optimizer_grouped_parameters.append(param_group)

        for group in self.args.custom_layer_parameters:
            layer_number = group.pop("layer")
            layer = f"layer.{layer_number}."
            group_d = {**group}
            group_nd = {**group}
            group_nd["weight_decay"] = 0.0
            params_d = []
            params_nd = []
            for n, p in model.named_parameters():
                if n not in custom_parameter_names and layer in n:
                    if any(nd in n for nd in no_decay):
                        params_nd.append(p)
                    else:
                        params_d.append(p)
                    custom_parameter_names.add(n)
            group_d["params"] = params_d
            group_nd["params"] = params_nd

            optimizer_grouped_parameters.append(group_d)
            optimizer_grouped_parameters.append(group_nd)

        if not self.args.train_custom_parameters_only:
            optimizer_grouped_parameters.extend(
                [
                    {
                        "params": [
                            p
                            for n, p in model.named_parameters()
                            if n not in custom_parameter_names and not any(nd in n for nd in no_decay)
                        ],
                        "weight_decay": args.weight_decay,
                    },
                    {
                        "params": [
                            p
                            for n, p in model.named_parameters()
                            if n not in custom_parameter_names and any(nd in n for nd in no_decay)
                        ],
                        "weight_decay": 0.0,
                    },
                ]
            )

        warmup_steps = math.ceil(t_total * args.warmup_ratio)
        args.warmup_steps = warmup_steps if args.warmup_steps == 0 else args.warmup_steps

        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon,)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
        )

        if args.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        global_step = 0
        tr_loss, logging_loss = 0.0, 0.0
        model.zero_grad()
        train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=args.silent, mininterval=0)
        epoch_number = 0
        best_eval_metric = None
        early_stopping_counter = 0
        steps_trained_in_current_epoch = 0
        epochs_trained = 0

        if args.model_name and os.path.exists(args.model_name):
            try:
                # set global_step to gobal_step of last saved checkpoint from model path
                checkpoint_suffix = args.model_name.split("/")[-1].split("-")
                if len(checkpoint_suffix) > 2:
                    checkpoint_suffix = checkpoint_suffix[1]
                else:
                    checkpoint_suffix = checkpoint_suffix[-1]
                global_step = int(checkpoint_suffix)
                epochs_trained = global_step // (len(train_dataloader) // args.gradient_accumulation_steps)
                steps_trained_in_current_epoch = global_step % (
                    len(train_dataloader) // args.gradient_accumulation_steps
                )

                logger.info("   Continuing training from checkpoint, will skip to saved global_step")
                logger.info("   Continuing training from epoch %d", epochs_trained)
                logger.info("   Continuing training from global step %d", global_step)
                logger.info("   Will skip the first %d steps in the current epoch", steps_trained_in_current_epoch)
            except ValueError:
                logger.info("   Starting fine-tuning.")

        if args.evaluate_during_training:
            training_progress_scores = self._create_training_progress_scores(**kwargs)
        if args.wandb_project:
            wandb.init(project=args.wandb_project, config={**asdict(args)}, **args.wandb_kwargs)
            wandb.watch(self.model)

        if args.fp16:
            from torch.cuda import amp

            scaler = amp.GradScaler()

        model.train()
        model_teacher.eval()
        for _ in train_iterator:
            if epochs_trained > 0:
                epochs_trained -= 1
                continue
            train_iterator.set_description(f"Epoch {epoch_number + 1} of {args.num_train_epochs}")
            batch_iterator = tqdm(
                train_dataloader,
                desc=f"Running Epoch {epoch_number} of {args.num_train_epochs}",
                disable=args.silent,
                mininterval=0,
            )
            for step, batch in enumerate(batch_iterator):
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    continue
                batch = tuple(t.to(device) for t in batch)

                inputs = self._get_inputs_dict(batch)

                labels = inputs.pop("labels")
                loss_distil = KLDivLoss(reduction="batchmean")
                if args.fp16:
                    with amp.autocast():
                        outputs_teacher = model_teacher(**inputs)
                        outputs = model(**inputs)
                        cls_hidden_state = outputs[2][-1][:, 0, :]
                        # model outputs are always tuple in pytorch-transformers (see doc)
                        logits = outputs[1]
                        logits_teacher = outputs_teacher[1]
                else:
                    outputs_teacher = model_teacher(**inputs)
                    outputs = model(**inputs)
                    # model outputs are always tuple in pytorch-transformers (see doc)
                    logits = outputs[1]
                    logits_teacher = outputs_teacher[1]
                    cls_hidden_state = outputs[2][-1][:, 0, :]
                # loss_multitask = f(cls_hidden_state)  This needs to be completed
                loss = loss_distil(F.log_softmax(logits / args.temperature, dim=-1), F.softmax(logits_teacher / args.temperature, dim=-1),) + args.alpha * loss_multitask
                if args.n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu parallel training

                current_loss = loss.item()

                if show_running_loss:
                    batch_iterator.set_description(
                        f"Epochs {epoch_number}/{args.num_train_epochs}. Running Loss: {current_loss:9.4f}"
                    )

                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                if args.fp16:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                tr_loss += loss.item()
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    if args.fp16:
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                    if args.fp16:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    scheduler.step()  # Update learning rate schedule
                    model.zero_grad()
                    global_step += 1

                    if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                        # Log metrics
                        tb_writer.add_scalar("lr", scheduler.get_lr()[0], global_step)
                        tb_writer.add_scalar(
                            "loss", (tr_loss - logging_loss) / args.logging_steps, global_step,
                        )
                        logging_loss = tr_loss
                        if args.wandb_project:
                            wandb.log(
                                {
                                    "Training loss": current_loss,
                                    "lr": scheduler.get_lr()[0],
                                    "global_step": global_step,
                                }
                            )

                    if args.save_steps > 0 and global_step % args.save_steps == 0:
                        # Save model checkpoint
                        output_dir_current = os.path.join(output_dir, "checkpoint-{}".format(global_step))

                        self.save_model(output_dir_current, optimizer, scheduler, model=model)

                    if args.evaluate_during_training and (
                        args.evaluate_during_training_steps > 0
                        and global_step % args.evaluate_during_training_steps == 0
                    ):

                        output_dir_current = os.path.join(output_dir, "checkpoint-{}".format(global_step))

                        os.makedirs(output_dir_current, exist_ok=True)

                        # Only evaluate when single GPU otherwise metrics may not average well
                        results, _, _ = self.eval_model(
                            eval_data,
                            verbose=verbose and args.evaluate_during_training_verbose,
                            wandb_log=False,
                            output_dir=output_dir_current,
                            **kwargs,
                        )
                        for key, value in results.items():
                            tb_writer.add_scalar("eval_{}".format(key), value, global_step)

                        if args.save_eval_checkpoints:
                            self.save_model(output_dir_current, optimizer, scheduler, model=model, results=results)

                        training_progress_scores["global_step"].append(global_step)
                        training_progress_scores["train_loss"].append(current_loss)
                        for key in results:
                            training_progress_scores[key].append(results[key])
                        report = pd.DataFrame(training_progress_scores)
                        report.to_csv(
                            os.path.join(args.output_dir, "training_progress_scores.csv"), index=False,
                        )

                        if args.wandb_project:
                            wandb.log(self._get_last_metrics(training_progress_scores))

                        if not best_eval_metric:
                            best_eval_metric = results[args.early_stopping_metric]
                            self.save_model(args.best_model_dir, optimizer, scheduler, model=model, results=results)
                        if best_eval_metric and args.early_stopping_metric_minimize:
                            if results[args.early_stopping_metric] - best_eval_metric < args.early_stopping_delta:
                                best_eval_metric = results[args.early_stopping_metric]
                                self.save_model(
                                    args.best_model_dir, optimizer, scheduler, model=model, results=results
                                )
                                early_stopping_counter = 0
                            else:
                                if args.use_early_stopping:
                                    if early_stopping_counter < args.early_stopping_patience:
                                        early_stopping_counter += 1
                                        if verbose:
                                            logger.info(f" No improvement in {args.early_stopping_metric}")
                                            logger.info(f" Current step: {early_stopping_counter}")
                                            logger.info(f" Early stopping patience: {args.early_stopping_patience}")
                                    else:
                                        if verbose:
                                            logger.info(f" Patience of {args.early_stopping_patience} steps reached")
                                            logger.info(" Training terminated.")
                                            train_iterator.close()
                                        return global_step, tr_loss / global_step
                        else:
                            if results[args.early_stopping_metric] - best_eval_metric > args.early_stopping_delta:
                                best_eval_metric = results[args.early_stopping_metric]
                                self.save_model(
                                    args.best_model_dir, optimizer, scheduler, model=model, results=results
                                )
                                early_stopping_counter = 0
                            else:
                                if args.use_early_stopping:
                                    if early_stopping_counter < args.early_stopping_patience:
                                        early_stopping_counter += 1
                                        if verbose:
                                            logger.info(f" No improvement in {args.early_stopping_metric}")
                                            logger.info(f" Current step: {early_stopping_counter}")
                                            logger.info(f" Early stopping patience: {args.early_stopping_patience}")
                                    else:
                                        if verbose:
                                            logger.info(f" Patience of {args.early_stopping_patience} steps reached")
                                            logger.info(" Training terminated.")
                                            train_iterator.close()
                                        return global_step, tr_loss / global_step

            epoch_number += 1
            output_dir_current = os.path.join(output_dir, "checkpoint-{}-epoch-{}".format(global_step, epoch_number))

            if args.save_model_every_epoch or args.evaluate_during_training:
                os.makedirs(output_dir_current, exist_ok=True)

            if args.save_model_every_epoch:
                self.save_model(output_dir_current, optimizer, scheduler, model=model)

            if args.evaluate_during_training:
                results, _, _ = self.eval_model(
                    eval_data, verbose=verbose and args.evaluate_during_training_verbose, wandb_log=False, **kwargs
                )

                self.save_model(output_dir_current, optimizer, scheduler, results=results)

                training_progress_scores["global_step"].append(global_step)
                training_progress_scores["train_loss"].append(current_loss)
                for key in results:
                    training_progress_scores[key].append(results[key])
                report = pd.DataFrame(training_progress_scores)
                report.to_csv(os.path.join(args.output_dir, "training_progress_scores.csv"), index=False)

                if args.wandb_project:
                    wandb.log(self._get_last_metrics(training_progress_scores))

                if not best_eval_metric:
                    best_eval_metric = results[args.early_stopping_metric]
                    self.save_model(args.best_model_dir, optimizer, scheduler, model=model, results=results)
                if best_eval_metric and args.early_stopping_metric_minimize:
                    if results[args.early_stopping_metric] - best_eval_metric < args.early_stopping_delta:
                        best_eval_metric = results[args.early_stopping_metric]
                        self.save_model(args.best_model_dir, optimizer, scheduler, model=model, results=results)
                        early_stopping_counter = 0
                    else:
                        if args.use_early_stopping and args.early_stopping_consider_epochs:
                            if early_stopping_counter < args.early_stopping_patience:
                                early_stopping_counter += 1
                                if verbose:
                                    logger.info(f" No improvement in {args.early_stopping_metric}")
                                    logger.info(f" Current step: {early_stopping_counter}")
                                    logger.info(f" Early stopping patience: {args.early_stopping_patience}")
                            else:
                                if verbose:
                                    logger.info(f" Patience of {args.early_stopping_patience} steps reached")
                                    logger.info(" Training terminated.")
                                    train_iterator.close()
                                return global_step, tr_loss / global_step
                else:
                    if results[args.early_stopping_metric] - best_eval_metric > args.early_stopping_delta:
                        best_eval_metric = results[args.early_stopping_metric]
                        self.save_model(args.best_model_dir, optimizer, scheduler, model=model, results=results)
                        early_stopping_counter = 0
                        early_stopping_counter = 0
                    else:
                        if args.use_early_stopping and args.early_stopping_consider_epochs:
                            if early_stopping_counter < args.early_stopping_patience:
                                early_stopping_counter += 1
                                if verbose:
                                    logger.info(f" No improvement in {args.early_stopping_metric}")
                                    logger.info(f" Current step: {early_stopping_counter}")
                                    logger.info(f" Early stopping patience: {args.early_stopping_patience}")
                            else:
                                if verbose:
                                    logger.info(f" Patience of {args.early_stopping_patience} steps reached")
                                    logger.info(" Training terminated.")
                                    train_iterator.close()
                                return global_step, tr_loss / global_step

        return global_step, tr_loss / global_step

    def eval_model(self, eval_data, output_dir=None, verbose=True, silent=False, wandb_log=True, **kwargs):
    ### This needs to be completed

    def evaluate(self, eval_dataset, output_dir, verbose=True, silent=False, wandb_log=True, **kwargs):
    ### This needs to be completed

    def predict(self, to_predict, split_on_space=True):
    ### This needs to be completed

    def load_and_cache_examples(self, data, evaluate=False, no_cache=False, to_predict=None):
    ### This needs to be completed

    def _move_model_to_device(self):
        self.model.to(self.device)

    def _get_last_metrics(self, metric_values):
        return {metric: values[-1] for metric, values in metric_values.items()}

    def _get_inputs_dict(self, batch):
        inputs = {
            "input_ids": batch[0],
            "attention_mask": batch[1],
            "labels": batch[3],
        }
        # XLM, DistilBERT, RoBERTa, and XLM-RoBERTa don't use segment_ids
        if self.args.model_type in ["bert", "xlnet", "albert"]:
            inputs["token_type_ids"] = batch[2]

        return inputs

    def _create_training_progress_scores(self, **kwargs):
        extra_metrics = {key: [] for key in kwargs}
        training_progress_scores = {
            "global_step": [],
            "precision": [],
            "recall": [],
            "f1_score": [],
            "train_loss": [],
            "eval_loss": [],
            **extra_metrics,
        }

        return training_progress_scores

    def save_model(self, output_dir=None, optimizer=None, scheduler=None, model=None, results=None):
        if not output_dir:
            output_dir = self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)

        if model and not self.args.no_save:
            # Take care of distributed/parallel training
            model_to_save = model.module if hasattr(model, "module") else model
            model_to_save.save_pretrained(output_dir)
            self.tokenizer.save_pretrained(output_dir)
            torch.save(self.args, os.path.join(output_dir, "training_args.bin"))
            if optimizer and scheduler and self.args.save_optimizer_and_scheduler:
                torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
            self._save_model_args(output_dir)

        if results:
            output_eval_file = os.path.join(output_dir, "eval_results.txt")
            with open(output_eval_file, "w") as writer:
                for key in sorted(results.keys()):
                    writer.write("{} = {}\n".format(key, str(results[key])))

    def _save_model_args(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        self.args.save(output_dir)

    def _load_model_args(self, input_dir):
        args = NERArgs()
        args.load(input_dir)
        return args

    def get_named_parameters(self):
        return [n for n, p in self.model.named_parameters()]
