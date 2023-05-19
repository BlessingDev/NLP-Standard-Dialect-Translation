from torch.utils.tensorboard import SummaryWriter
import json
import os

scalar_keys = [
    "loss/train",
    "loss/val"
]

def log_file_to_tensorboard(log_dir, file_path):
    json_dict = {}
    with open(file_path, mode="r+", encoding="utf-8") as fp:
        json_dict = json.loads(fp.read())
    
    tb_file_name = "emb={emb},hidden={rh},decay={dec}".format(
        emb=json_dict["embedding_size"],
        rh=json_dict["rnn_hidden_size"],
        dec=json_dict["opt_weight_decay"]
    )

    writer = SummaryWriter(log_dir=os.path.join(log_dir, tb_file_name))
    
    for s_k in scalar_keys:
        scalar_list:list = json_dict.get(s_k, [])

        for i, s in enumerate(scalar_list):
            writer.add_scalar(s_k, s, i)
    
    writer.flush()

    writer.close()

if __name__ == "__main__":
    log_file_to_tensorboard("model_storage/labeling_model_2/tb_logs", "model_storage/labeling_model_2/logs/train_at_2023-05-09_07_46.json")