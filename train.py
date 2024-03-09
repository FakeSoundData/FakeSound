import os
import json
import librosa
import numpy as np
import pandas as pd
import random
import argparse
from tqdm.auto import tqdm
from datetime import datetime
import torch

import soundfile as sf
import torchaudio
from torch.utils.data import Dataset, DataLoader
from transformers import SchedulerType, get_scheduler

import sys
WORKSPACE_PATH = "WORKSPACE_PATH"
sys.path.extend([WORKSPACE_PATH])

from models import detection_model
def parse_args():
    parser = argparse.ArgumentParser(description="Train a deepfake audio detection model.")
    parser.add_argument(
        "--train_file", '-f', type=str, default=f"{WORKSPACE_PATH}/deepfake_data/ldm2_train.json"
    )
    parser.add_argument(
        "--batch_size", '-b', type=int, default=16,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--learning_rate", '-lr', type=float, default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--num_epochs", '-e', type=int, default=40,
        help="Total number of training epochs to perform."
    )
    parser.add_argument(
        "--output_dir", '-o', type=str, default=None,
        help="Where to store the final model."
    )
    parser.add_argument(
        "--model_class", '-m', type=str, default="WavLM_Detection",
        help="name of model_class"
    )
    parser.add_argument(
        "--multi_task", '-mt', default=False, action='store_true',
        help="multi_task"
    )
    parser.add_argument(
        "--multi_task_ratio", '-mtr', default=0.1, type=float,
        help="multi_task"
    )
    parser.add_argument(
        "--duration", '-d', type=float, default=10,
        help="Audio duration."
    )
    parser.add_argument(
        "--time_resolution", type=float, default=0.02,
        help="."
    )
    parser.add_argument(
        "--sample_rate", type=int, default=16000,
        help="."
    )
    parser.add_argument(
        "--num_examples", '-n', type=int, default=-1,
        help="How many examples to use for training.",
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="A seed for reproducible training."
    )
    args = parser.parse_args()

    return args


class DeepfakeDetectionDataset(Dataset):
    def __init__(self, data_file, args):
        
        self.data = json.load(open(data_file, "r"))["audios"]
        self.time_resolution = args.time_resolution
        self.duration = args.duration
        self.sample_rate = args.sample_rate
        num_examples = args.num_examples
        if num_examples != -1:
            self.data = self.data[:num_examples]
        self.model_class = args.model_class

    def __len__(self):
        return len(self.data)

    def _load_wav(self, source_file):
        assert source_file.endswith('.wav'), "the standard format of file should be '.wav' "
        wav, sr = sf.read(source_file)
        channel = sf.info(source_file).channels
        assert channel == 1, "Channel should be 1, but got {} in file {}".format(channel, source_file)
        source = torch.from_numpy(wav).float()
        if sr != 16e3: 
            source = torchaudio.functional.resample(source, orig_freq=sr, new_freq=self.sample_rate).float()  

        if self.model_class == "WavLM_Detection":
            # [N, 160080] in batch
            # audio, sr = librosa.load(source_file, sr=self.sample_rate)
            # audio = np.concatenate((audio, np.zeros(80)), axis=0) # for WavLM alignment          
            source = torch.cat((source, torch.zeros(80)), axis=0)
        else:
            # [N, 1, 1024, 128] in batch
            assert self.model_class == "EAT_Detection"   
            target_length, norm_mean, norm_std  = 1024, -4.268,  4.569  
            
            source = source - source.mean()
            source = source.unsqueeze(dim=0)
            source = torchaudio.compliance.kaldi.fbank(source, htk_compat=True, sample_frequency=16000, use_energy=False,
                window_type='hanning', num_mel_bins=128, dither=0.0, frame_shift=10).unsqueeze(dim=0)
            
            n_frames = source.shape[1]
            diff = target_length - n_frames
            if diff > 0:
                m = torch.nn.ZeroPad2d((0, 0, 0, diff)) 
                source = m(source)
                
            elif diff < 0:
                source = source[0:target_length, :]
                        
            source = (source - norm_mean) / (norm_std * 2)
        return source.numpy()

    def __getitem__(self, index):
        item = self.data[index]

        audio = self._load_wav(item["filepath"])

        binary_label = int(item["label"]) # fake->1, real->0
        tgt = np.zeros(int(self.duration / self.time_resolution))
        if binary_label == 1:
            [onset, offset] = item["onset_offset"].split("_")
            tgt[int(float(onset) / self.time_resolution): int(float(offset) / self.time_resolution)] = 1

        return audio, binary_label, tgt, item["audio_id"]

    def collate_fn(self, data):
        dat = pd.DataFrame(data)
        batch = []
        for i in dat:
            if i==3:
                batch.append(dat[i].tolist())
            elif i==1:
                batch.append(np.array(dat[i]))
            else:
                batch.append(torch.tensor(np.array(dat[i].tolist()), dtype=torch.float32))   
        return batch
    
    
def main():
    args = parse_args()
    print(args)
    # If passed along, set the training seed now.
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)    
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Handle output directory creation
    if args.output_dir is None or args.output_dir == "":
        args.output_dir = f"{WORKSPACE_PATH}/ckpts/{args.model_class}/multi-task-{args.multi_task}_ldm2_trainth1-4"        
    elif args.output_dir is not None:
        args.output_dir = f"{WORKSPACE_PATH}/ckpts/{args.model_class}/multi-task-{args.multi_task}_{args.output_dir}"
    os.makedirs(args.output_dir, exist_ok=True)
    with open("{}/summary.jsonl".format(args.output_dir), "w") as f:
        f.write(json.dumps(dict(vars(args))) + "\n\n")

    # Init
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = getattr(detection_model, args.model_class)(multi_task=args.multi_task).to(device)
    train_dataset = DeepfakeDetectionDataset(args.train_file, args)       
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size, collate_fn=train_dataset.collate_fn)
    print(F"Num instances in train: {len(train_dataset)}")

    # Optimizer
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer_parameters = model.parameters()
    if hasattr(model, "future_extractor"):
        for param in model.future_extractor.parameters():
            param.requires_grad = False
            model.future_extractor.eval()
            optimizer_parameters = model.backbone.parameters()
        print("Optimizing backbone parameters.")
    num_trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Num trainable parameters: {}".format(num_trainable_parameters))
    optimizer = torch.optim.AdamW(
        optimizer_parameters, lr=args.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=2e-4,
        eps=1e-08,
    )

    num_update_steps_per_epoch = len(train_dataloader)
    args.max_train_steps = args.num_epochs * num_update_steps_per_epoch
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=100,
        num_training_steps=args.max_train_steps,
    )

    # Train!
    total_batch_size = args.batch_size
    print("***** Running training *****")
    print(f"  Num examples = {len(train_dataset)}")
    print(f"  Num Epochs = {args.num_epochs}")
    print(f"  Instantaneous batch size per device = {args.batch_size}")
    print(f"  Total optimization steps = {args.max_train_steps}")

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps))

    completed_steps = 0
    starting_epoch = 0  
    # Duration of the audio clips in seconds
    duration, best_loss, best_epoch = args.duration, np.inf, 0

    for epoch in range(starting_epoch, args.num_epochs):
        model.train()
        total_loss = 0
        print(f"train epoch {epoch} begin!")
        for step, batch in enumerate(train_dataloader):
            audio, binary_label, tgt, _ = batch # audio, binary_label, tgt, audio_id
            output = model(audio.to(device))
            loss = criterion(output["pred"].squeeze(-1), tgt.to(device))
            if hasattr(model, "multi_task_classifier"):
                loss = (1.0 - args.multi_task_ratio) * loss + args.multi_task_ratio * criterion(output["pred_binary"].squeeze(-1), torch.tensor(binary_label, dtype=float).to(device))
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            total_loss += loss.detach().float()
            progress_bar.update(1)
            completed_steps += 1
        print(f"train epoch {epoch} finish!")
 
        result = {}
        result["epoch"] = epoch,
        result["step"] = completed_steps
        result["train_loss"] = round(total_loss.item()/len(train_dataloader), 4)

        if result["train_loss"] < best_loss:
            best_loss = result["train_loss"]
            best_epoch = epoch
            torch.save(model.state_dict(), f"{args.output_dir}/best.pt")
        if epoch > 0 and epoch % 10 == 0:
            torch.save(model.state_dict(), f"{args.output_dir}/epoch{epoch}.pt")
        result["best_eopch"] = best_epoch
        print(result)
        result["time"] = datetime.now().strftime("%y-%m-%d-%H-%M-%S")

        with open("{}/summary.jsonl".format(args.output_dir), "a") as f:
            f.write(json.dumps(result) + "\n\n")
  
            
if __name__ == "__main__":
    main()
