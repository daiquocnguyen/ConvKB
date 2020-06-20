from Config import Config
from ConvKB import ConvKB
import json
import os
import numpy as np

from argparse import ArgumentParser
parser = ArgumentParser("ConvKB")
parser.add_argument("--dataset", default="WN18RR", help="Name of the dataset.")
parser.add_argument("--learning_rate", default=0.01, type=float, help="Learning rate")
parser.add_argument("--nbatches", default=100, type=int, help="Number of batches")
parser.add_argument("--num_epochs", default=300, type=int, help="Number of training epochs")
parser.add_argument("--model_name", default='WN18RR', help="")
parser.add_argument('--neg_num', default=10, type=int, help='')
parser.add_argument('--hidden_size', type=int, default=50, help='')
parser.add_argument('--num_of_filters', type=int, default=64, help='')
parser.add_argument('--dropout', type=float, default=0.5, help='')
parser.add_argument('--save_steps', type=int, default=1000, help='')
parser.add_argument('--valid_steps', type=int, default=50, help='')
parser.add_argument("--lmbda", default=0.2, type=float, help="")
parser.add_argument("--lmbda2", default=0.01, type=float, help="")
parser.add_argument("--mode", choices=["train", "predict"], default="train", type=str)
parser.add_argument("--checkpoint_path", default=None, type=str)
parser.add_argument("--test_file", default="", type=str)
parser.add_argument("--optim", default='adagrad', help="")
parser.add_argument('--use_init', default=1, type=int, help='')
parser.add_argument('--kernel_size', default=1, type=int, help='')
args = parser.parse_args()

print(args)

out_dir = os.path.abspath(os.path.join("../runs_pytorch_ConvKB/"))
print("Writing to {}\n".format(out_dir))
# Checkpoint directory
checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
result_dir = os.path.abspath(os.path.join(checkpoint_dir, args.model_name))
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
if not os.path.exists(result_dir):
    os.makedirs(result_dir)

con = Config()
in_path = "./benchmarks/" + args.dataset + "/"
con.set_in_path(in_path)
test_file_path = ""
if args.test_file != "":
    test_file_path = in_path + args.test_file
con.set_test_file_path(test_file_path)
con.set_work_threads(8)
con.set_train_times(args.num_epochs)
con.set_nbatches(args.nbatches)
con.set_alpha(args.learning_rate)
con.set_bern(1)
con.set_dimension(args.hidden_size)
con.set_lmbda(args.lmbda)
con.set_lmbda_two(0.01)
con.set_margin(1.0)
con.set_ent_neg_rate(args.neg_num)
con.set_opt_method(args.optim)
con.set_save_steps(args.save_steps)
con.set_valid_steps(args.valid_steps)
con.set_early_stopping_patience(10)
con.set_checkpoint_dir(checkpoint_dir)
con.set_result_dir(result_dir)
# set knowledge graph completion
con.set_test_link(True)
con.init()

def get_term_id(filename):
    entity2id = {}
    id2entity = {}
    with open(filename) as f:
        for line in f:
            if len(line.strip().split()) > 1:
                tmp = line.strip().split()
                entity2id[tmp[0]] = int(tmp[1])
                id2entity[int(tmp[1])] = tmp[0]
    return entity2id, id2entity

def get_init_embeddings(relinit, entinit):
    lstent = []
    lstrel = []
    with open(relinit) as f:
        for line in f:
            tmp = [float(val) for val in line.strip().split()]
            lstrel.append(tmp)
    with open(entinit) as f:
        for line in f:
            tmp = [float(val) for val in line.strip().split()]
            lstent.append(tmp)
    return np.array(lstent, dtype=np.float32), np.array(lstrel, dtype=np.float32)


if args.mode == "train":

    if args.use_init:
        hidden_size = "100"  # for FB15K-237
        con.set_dimension(100)
        if args.dataset == "WN18RR":
            hidden_size = "50"
            con.set_dimension(50)

        init_entity_embs, init_relation_embs = get_init_embeddings(
            "./benchmarks/" + args.dataset + "/relation2vec"+hidden_size+".init",
            "./benchmarks/" + args.dataset + "/entity2vec"+hidden_size+".init")

        e2id, id2e = get_term_id(filename="./benchmarks/" + args.dataset + "/entity2id.txt")
        e2id50, id2e50 = get_term_id(filename="./benchmarks/" + args.dataset + "/entity2id_"+hidden_size+"init.txt")
        assert len(e2id) == len(e2id50)

        entity_embs = np.empty([len(e2id), con.hidden_size]).astype(np.float32)
        for i in range(len(e2id)):
            _word = id2e[i]
            id = e2id50[_word]
            entity_embs[i] = init_entity_embs[id]

        r2id, id2r = get_term_id(filename="./benchmarks/" + args.dataset + "/relation2id.txt")
        r2id50, id2r50 = get_term_id(filename="./benchmarks/" + args.dataset + "/relation2id_"+hidden_size+"init.txt")
        assert len(r2id) == len(r2id50)

        rel_embs = np.empty([len(r2id), con.hidden_size]).astype(np.float32)
        for i in range(len(r2id)):
            _rel = id2r[i]
            id = r2id50[_rel]
            rel_embs[i] = init_relation_embs[id]

        con.set_init_embeddings(entity_embs, rel_embs)

    con.set_config_CNN(num_of_filters=args.num_of_filters, drop_prob=args.dropout, kernel_size=args.kernel_size)
    con.set_train_model(ConvKB)
    con.training_model()

else:
    if args.use_init:
        hidden_size = "100"  # for FB15K-237
        con.set_dimension(100)
        if args.dataset == "WN18RR":
            hidden_size = "50"
            con.set_dimension(50)

    con.set_config_CNN(num_of_filters=args.num_of_filters, drop_prob=args.dropout, kernel_size=args.kernel_size)
    con.set_test_model(ConvKB, args.checkpoint_path)
    con.test()

