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
"""TrainOneStepWithLossScaleCellGlobalNormClip"""

import mindspore

from mindspore import Tensor, Parameter
from mindspore import ops

import mindspore.nn as nn
from mindspore.common import RowTensor
from mindspore.ops import composite as C
from mindspore.ops import functional as F
from mindspore.ops import operations as P
from mindspore.common.parameter import ParameterTuple

_grad_scale = C.MultitypeFuncGraph("grad_scale")
reciprocal = P.Reciprocal()


@_grad_scale.register("Tensor", "Tensor")
def tensor_grad_scale(scale, grad):
    return grad * F.cast(reciprocal(scale), F.dtype(grad))


@_grad_scale.register("Tensor", "RowTensor")
def tensor_grad_scale_row_tensor(scale, grad):
    return RowTensor(grad.indices,
                     grad.values * F.cast(reciprocal(scale), F.dtype(grad.values)),
                     grad.dense_shape)


_grad_overflow = C.MultitypeFuncGraph("_grad_overflow")
grad_overflow = P.FloatStatus()


class TrainOneStepWithLossScaleCellGlobalNormClip(nn.TrainOneStepWithLossScaleCell):
    """
    Encapsulation class of SSD network training.

    Append an optimizer to the training network after that the construct
    function can be called to create the backward graph.

    Args:
        network (Cell): The training network. Note that loss function should have been added.
        optimizer (Optimizer): Optimizer for updating the weights.
        sens (Number): The adjust parameter. Default: 1.0.
        use_global_nrom(bool): Whether apply global norm before optimizer. Default: False
    """

    def __init__(self, network, optimizer,
                 scale_sense=1.0, use_global_norm=True,
                 clip_global_norm_value=1.0,
                 ema = True, 
                 decay=0.9998, 
                 updates=0,
                 moving_name=None,
                 ema_moving_weight=None
                 ):

        super(TrainOneStepWithLossScaleCellGlobalNormClip, self).__init__(network, optimizer, scale_sense)
        self.use_global_norm = use_global_norm
        self.clip_global_norm_value = clip_global_norm_value
        self.print = P.Print()

        self.ema = ema
        self.moving_name = moving_name
        self.ema_moving_weight = ema_moving_weight
        if self.ema:
            print("====================create EMA================")
            self.ema_weight = self.weights.clone("ema", init='same')
            self.decay = decay
            self.updates = Parameter(Tensor(updates, mindspore.float32))
            self.assign = ops.Assign()
            self.ema_moving_parameters()

    def ema_moving_parameters(self):
        self.moving_name = {}
        moving_list = []
        idx = 0
        for key, param in self.network.parameters_and_names():
            if "moving_mean" in key or "moving_variance" in key:
                new_param = param.clone()
                new_param.name = "ema." + param.name
                moving_list.append(new_param)
                self.moving_name["ema." + key] = idx
                idx += 1
        self.ema_moving_weight = ParameterTuple(moving_list)


    def ema_update(self):
        """
        Update EMA parameters.
        """
        if self.ema:
            self.updates += 1
            # d = self.decay * (1 - ops.Exp()(-self.updates / 2000))
            d = self.decay

            # update trainable parameters
            for ema_v, weight in zip(self.ema_weight, self.weights):
                tep_v = ema_v * d
                self.assign(ema_v, (1.0 - d) * weight + tep_v)
        return self.updates

    # moving_parameter_update is executed inside the callback(EMACallBack)
    def moving_parameter_update(self):
        if self.ema:
            # d = (self.decay * (1 - ops.Exp()(-self.updates / 2000))).asnumpy().item()
            d = self.decay
            # update moving mean and moving var
            for key, param in self.network.parameters_and_names():
                if "moving_mean" in key or "moving_variance" in key:
                    idx = self.moving_name["ema." + key]
                    moving_weight = param.asnumpy()
                    tep_v = self.ema_moving_weight[idx] * d
                    ema_value = (1.0 - d) * moving_weight + tep_v
                    self.ema_moving_weight[idx] = ema_value

    def construct(self, *inputs):
        """
        construct
        """
        weights = self.weights
        loss = self.network(*inputs)
        scaling_sens = self.scale_sense

        status, scaling_sens = self.start_overflow_check(loss, scaling_sens)

        scaling_sens_filled = C.ones_like(loss) * F.cast(scaling_sens, F.dtype(loss))
        grads = self.grad(self.network, weights)(*inputs, scaling_sens_filled)
        grads = self.hyper_map(F.partial(_grad_scale, scaling_sens), grads)
        # apply grad reducer on grads
        grads = self.grad_reducer(grads)
        self.ema_update()

        # get the overflow buffer
        cond = self.get_overflow_status(status, grads)
        overflow = self.process_loss_scale(cond)
        # if there is no overflow, do optimize
        if not overflow:
            if self.use_global_norm:
                grads = C.clip_by_global_norm(grads, clip_norm=self.clip_global_norm_value)
            loss = F.depend(loss, self.optimizer(grads))
        else:
            self.print("=============Over Flow, skipping=============")
        return loss
