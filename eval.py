# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""train"""
import os
import time

from mindspore import Model
from mindspore import context
from mindspore import nn
from mindspore import ops
from mindspore.common import set_seed
from mindspore import load_checkpoint, load_param_into_net


from src.args import args
from src.tools.callback import EvaluateCallBack,VarMonitor
from src.tools.cell import cast_amp
from src.tools.criterion import get_criterion, NetWithLoss
from src.tools.get_misc import get_dataset, set_device, get_model, pretrained, get_train_one_step
from src.tools.optimizer import get_optimizer


def main():
    assert args.crop, f"{args.arch} is only for evaluation"
    set_seed(args.seed)
    mode = {
        0: context.GRAPH_MODE,
        1: context.PYNATIVE_MODE
    }
    context.set_context(mode=mode[args.graph_mode], device_target=args.device_target)
    context.set_context(enable_graph_kernel=False)
    if args.device_target == "Ascend":
        context.set_context(enable_auto_mixed_precision=True)
        
    rank = set_device(args)

    # get model and cast amp_level
    net = get_model(args)

    # load checkpoint
    # checkpoint_file_path = "./checkpoint/cswin_small_2240-304_1251.ckpt"
    
    param_dict = load_checkpoint(args.checkpoint_file_path)
    load_param_into_net(net, param_dict)

    print("model: ", args.checkpoint_file_path)
    cast_amp(net)
    criterion = get_criterion(args)
    net_with_loss = NetWithLoss(net, criterion)
    if args.pretrained:
        pretrained(args, net)

    data = get_dataset(args)

    batch_num = data.train_dataset.get_dataset_size()
    optimizer, _= get_optimizer(args, net, batch_num)
    # save a yaml file to read to record parameters

    net_with_loss = get_train_one_step(args, net_with_loss, optimizer)

    eval_network = nn.WithEvalCell(net, criterion, args.amp_level in ["O2", "O3", "auto"])
    eval_indexes = [0, 1, 2]
    model = Model(net_with_loss, metrics={'top_1_accuracy', 'top_5_accuracy', "loss"},  # "acc",'top_1_accuracy', 'top_5_accuracy'
                  eval_network=eval_network,
                  eval_indexes=eval_indexes)

    start = time.time()
    print("start {} start to eval" .format(time.asctime()))
    res = model.eval(data.val_dataset)
    print("result: ", res)

    end = time.time()
    print("eval time: ", end - start)



if __name__ == '__main__':
    main()

# python eval.py --config ./src/configs/small_cswin_224.yaml
