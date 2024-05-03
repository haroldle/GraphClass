import jax
import torch
import os
import equinox as eqx
import random
import jax.numpy as jnp
from jaxtyping import Array, Float, Int, PyTree
import optax
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image, ImageReadMode
import numpy as np


class RadioMapDataset(Dataset):
    def __init__(self, data_path, typeOfData, typeOfAntenna='antennas'):
        self.data_path = data_path
        self.typeOfData = typeOfData
        self.typeOfAntenna = typeOfAntenna
        self.list_images = os.listdir(self.data_path + '/' + typeOfData)

    def __len__(self):
        return len(self.list_images)

    def __getitem__(self, idx):
        antenna = '_'.join(self.list_images[idx].split('_')[:2]) + '.png'
        single_signal_path = self.data_path + '/Downloads/RadioMapSeer/gain/' + 'DPM' + '/' + antenna

        building = self.data_path + '/Downloads/RadioMapSeer/png/' + 'buildings_complete' + '/' + f'{self.list_images[idx].split("_")[0]}.png'
        mask = self.data_path + '/' + self.typeOfData + '/' + self.list_images[idx]
        target = self.data_path + '/Downloads/RadioMapSeer/png/' + self.typeOfAntenna + '/' + antenna

        single_signal_image = read_image(single_signal_path, ImageReadMode.GRAY).float()
        mask = read_image(mask, ImageReadMode.GRAY).float()
        target = read_image(target, ImageReadMode.GRAY).float() / 255

        building = read_image(building).float()

        single_signal_image /= 255.0

        building = 0.0 - building / 255.0
        sampling_pixel_img = mask + building

        data = torch.cat([single_signal_image * mask, sampling_pixel_img])
        return data, target


def collate_fn(batch_data):
    data, targets = [datum[0] for datum in batch_data], [datum[1] for datum in batch_data]
    return np.array(data), np.array(targets)


# https://github.com/WangYueFt/dgcnn/blob/master/pytorch/main.py
# EdgeConv
@jax.jit
def knn_jax(
        x: Float[Array, "n_dims row*col"]
):
    # X SHOULD BE C, H*W
    # SUM ALONG THE CHANNELS LIKE FROM N, H * W TO 1, H*W
    x2 = jnp.sum(x ** 2, axis=0, keepdims=True)
    # H*W, C CROSS C,H*W
    xy = jnp.matmul(x.transpose(1, 0), x)
    dist = -x2 + 2 * xy - x2.transpose(1, 0)
    _, idx = jax.lax.top_k(dist, k=20)
    return idx


@jax.jit
def get_graph_feature_jax(
        x: Float[Array, "C H W"]
):
    n_dims = x.shape[0]
    all_pixels = x.shape[1]
    x_feature_maps = jnp.reshape(x, (n_dims, all_pixels))
    idx = knn_jax(x_feature_maps)
    # Feature maps from => n_dims, row*col to row*col, n_dims and [idx] just add another dimension
    # We have row*col, 20, 3
    graph_feature_map = x_feature_maps.transpose(1, 0)[idx]

    # return a channels, row*col, k
    return graph_feature_map.transpose(2, 0, 1)


class GINModule(eqx.Module):
    gin_linear: eqx.nn.Conv2d
    leaky_relu_alpha: float
    eps: jax.numpy.array
    leaky_relu: jax.nn.leaky_relu

    def __init__(self, in_channels, out_channels, groups, leaky_relu_alpha, key):
        weight_init_key, *conv_key = jax.random.split(key, 1 + 2)
        self.gin_linear = eqx.nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            use_bias=False,
            key=conv_key[0]
        )
        self.eps = jnp.zeros(
            (in_channels, 1024, 1),
            dtype=jnp.float32
        )

        self.leaky_relu = jax.nn.leaky_relu
        self.leaky_relu_alpha = leaky_relu_alpha
        self._init_weights(weight_init_key)

    def _init_weights(self, key_: jax.Array):
        where = lambda l: l.weight
        new_weights_pool_linear = jax.nn.initializers.he_normal()(
            key_,
            self.gin_linear.weight.shape,
            dtype=jnp.float32
        )

        self.gin_linear = eqx.tree_at(
            where,
            self.gin_linear,
            new_weights_pool_linear
        )

    def __call__(self, x):
        # GET DENSE KNN FEATURE GRAPHS
        x_knn = get_graph_feature_jax(x)
        # AGGREGATE POOLING
        x_ = jnp.sum(
            x_knn,
            axis=-1,
            keepdims=True
        )
        return jnp.squeeze(
            self.gin_linear(
                (1.0 + self.eps) * jnp.expand_dims(x, axis=-1) + x_
            ),
            axis=-1
        )


class Encoder(eqx.Module):
    groupNorms: list
    conv_layers: list
    average_pooling_layers: list
    leaky_relu_alpha: float
    leaky_relu: jax.nn.leaky_relu
    graph_module: list

    def __init__(
            self,
            enc_in: int,
            enc_out: int,
            n_dim: int,
            leaky_relu_alpha: float,
            key: jax.Array
    ):
        new_key, *conv_keys = jax.random.split(
            key,
            1 + 10
        )
        mlp_key, *graph_keys = jax.random.split(conv_keys[-1], 1 + 5)
        self.groupNorms = [
            eqx.nn.GroupNorm(
                groups=9,
                channels=n_dim
            ) for _ in range(9)
        ]
        self.groupNorms.append(
            eqx.nn.GroupNorm(
                2,
                channels=enc_out
            )
        )

        self.conv_layers = [
                               eqx.nn.Conv2d(
                                   in_channels=enc_in,
                                   out_channels=n_dim,
                                   kernel_size=(3, 3),
                                   stride=1,
                                   padding=1,
                                   key=conv_keys[0]
                               )
                           ] + [
                               eqx.nn.Conv2d(
                                   in_channels=n_dim,
                                   out_channels=n_dim,
                                   kernel_size=(3, 3),
                                   stride=1,
                                   padding=1,
                                   key=conv_keys[i + 1]
                               ) for i in range(8)
                           ]

        self.graph_module = [
                                GINModule(
                                    in_channels=n_dim,
                                    out_channels=n_dim,
                                    groups=9,
                                    leaky_relu_alpha=leaky_relu_alpha,
                                    key=graph_keys[i]
                                ) for i in range(5)
                            ] + [
                                eqx.nn.Conv1d(
                                    in_channels=n_dim * 5,
                                    out_channels=4,
                                    kernel_size=1,
                                    key=mlp_key
                                )
                            ]

        self.average_pooling_layers = [
            eqx.nn.AvgPool2d(
                kernel_size=(2, 2),
                stride=2
            )
            for _ in range(3)
        ]

        self.leaky_relu_alpha = leaky_relu_alpha
        self.leaky_relu = jax.nn.leaky_relu
        _ = self._init_weights(key=new_key)

    def _init_weights(self, key):
        new_key, *weight_initializer_key = jax.random.split(key, 11)
        where = lambda l: l.weight
        for i, layer in enumerate(self.conv_layers[:-1]):
            new_weights = jax.nn.initializers.he_normal()(
                weight_initializer_key[i],
                layer.weight.shape,
                dtype=jnp.float32
            )

            self.conv_layers[i] = eqx.tree_at(
                where,
                layer,
                new_weights
            )
        return new_key

    def __call__(self, x: Float[Array, "2 256 256"]):
        x = self.leaky_relu(self.groupNorms[0](self.conv_layers[0](x)), self.leaky_relu_alpha)
        x = self.leaky_relu(self.groupNorms[1](self.conv_layers[1](x)), self.leaky_relu_alpha)
        x = self.leaky_relu(self.groupNorms[2](self.conv_layers[2](x)), self.leaky_relu_alpha)
        skip_1 = x
        x = self.average_pooling_layers[0](x)

        x = self.leaky_relu(self.groupNorms[3](self.conv_layers[3](x)), self.leaky_relu_alpha)
        x = self.leaky_relu(self.groupNorms[4](self.conv_layers[4](x)), self.leaky_relu_alpha)
        x = self.leaky_relu(self.groupNorms[5](self.conv_layers[5](x)), self.leaky_relu_alpha)
        skip_2 = x
        x = self.average_pooling_layers[1](x)

        x = self.leaky_relu(self.groupNorms[6](self.conv_layers[6](x)), self.leaky_relu_alpha)
        x = self.leaky_relu(self.groupNorms[7](self.conv_layers[7](x)), self.leaky_relu_alpha)
        x = self.leaky_relu(self.groupNorms[8](self.conv_layers[8](x)), self.leaky_relu_alpha)
        skip_3 = x
        x = self.average_pooling_layers[2](x)

        x = jnp.reshape(x, (x.shape[0], x.shape[1] * x.shape[2]))

        x1 = self.graph_module[0](x)
        x2 = self.graph_module[1](x1)
        x3 = self.graph_module[2](x2)
        x4 = self.graph_module[3](x3)
        x5 = self.graph_module[4](x4)

        x = self.leaky_relu(
            self.graph_module[-1](jnp.concatenate([x1, x2, x3, x4, x5], axis=0)),
            self.leaky_relu_alpha
        )
        x = jnp.reshape(x, newshape=(4, 32, 32))
        return x, skip_1, skip_2, skip_3


class Decoder(eqx.Module):
    groupNorms: list
    conv_transpose_layers: list
    leaky_relu_alpha: float
    leaky_relu: jax.nn.leaky_relu
    conv2d_output: eqx.nn.Conv2d

    def __init__(self,
                 dec_in: int,
                 dec_out: int,
                 n_dim: int,
                 leaky_relu_alpha: float,
                 key: jax.Array):
        new_key, conv_key, *conv_tranpose_keys = jax.random.split(
            key,
            1 + 1 + 10
        )

        self.groupNorms = [
                              eqx.nn.GroupNorm(
                                  2,
                                  channels=dec_in
                              )
                          ] + [
                              eqx.nn.GroupNorm(
                                  groups=9,
                                  channels=n_dim
                              ) for _ in range(9)
                          ]

        self.conv_transpose_layers = [
                                         eqx.nn.ConvTranspose2d(
                                             in_channels=dec_in,
                                             out_channels=dec_in,
                                             kernel_size=(3, 3),
                                             stride=1,
                                             padding=1,
                                             key=conv_tranpose_keys[0]
                                         ),
                                         eqx.nn.ConvTranspose2d(
                                             in_channels=dec_in + n_dim,
                                             out_channels=n_dim,
                                             kernel_size=(3, 3),
                                             stride=1,
                                             padding=1,
                                             key=conv_tranpose_keys[1]
                                         )
                                     ] + [
                                         eqx.nn.ConvTranspose2d(
                                             in_channels=n_dim,
                                             out_channels=n_dim,
                                             kernel_size=(3, 3),
                                             stride=1,
                                             padding=1,
                                             key=conv_tranpose_keys[i + 1]) if i % 3 != 2
                                         else eqx.nn.ConvTranspose2d(
                                             in_channels=2 * n_dim,
                                             out_channels=n_dim,
                                             kernel_size=(3, 3),
                                             stride=1,
                                             padding=1,
                                             key=conv_tranpose_keys[i + 1]
                                         )
                                         for i in range(8)
                                     ]

        self.conv2d_output = eqx.nn.Conv2d(
            n_dim,
            1,
            kernel_size=(1, 1),
            key=conv_key
        )

        self.leaky_relu_alpha = leaky_relu_alpha
        self.leaky_relu = jax.nn.leaky_relu
        self._init_weights(key=new_key)

    def _init_weights(self, key: jax.Array):
        new_key, conv_weight_initializer_key, *weight_initializer_key = jax.random.split(key, 1 + 1 + 10)
        _, *linear_weight_initializer_keys = jax.random.split(new_key, 1 + 3)
        where = lambda l: l.weight
        for i, layer in enumerate(self.conv_transpose_layers):
            new_weights = jax.nn.initializers.he_normal()(
                weight_initializer_key[i],
                layer.weight.shape,
                dtype=jnp.float32
            )

            self.conv_transpose_layers[i] = eqx.tree_at(
                where,
                layer,
                new_weights
            )

        new_conv_weights = jax.nn.initializers.he_normal()(
            conv_weight_initializer_key,
            self.conv2d_output.weight.shape,
            dtype=jnp.float32
        )
        self.conv2d_output = eqx.tree_at(where,
                                         self.conv2d_output,
                                         new_conv_weights)

    def __call__(self,
                 x: Float[Array, "4 32 32"],
                 skip_1: Float[Array, "27 256 256"],
                 skip_2: Float[Array, "27 128 128"],
                 skip_3: Float[Array, "27 64 64"],
                 ):
        x = self.leaky_relu(self.groupNorms[0](self.conv_transpose_layers[0](x)), self.leaky_relu_alpha)
        x = jax.image.resize(image=x, shape=(x.shape[0], x.shape[1] * 2, x.shape[2] * 2), method="bilinear")
        x = jnp.concatenate([x, skip_3], axis=0)

        x = self.leaky_relu(self.groupNorms[1](self.conv_transpose_layers[1](x)), self.leaky_relu_alpha)
        x = self.leaky_relu(self.groupNorms[2](self.conv_transpose_layers[2](x)), self.leaky_relu_alpha)
        x = self.leaky_relu(self.groupNorms[3](self.conv_transpose_layers[3](x)), self.leaky_relu_alpha)

        x = jax.image.resize(image=x, shape=(x.shape[0], x.shape[1] * 2, x.shape[2] * 2), method="bilinear")
        x = jnp.concatenate([x, skip_2], axis=0)

        x = self.leaky_relu(self.groupNorms[4](self.conv_transpose_layers[4](x)), self.leaky_relu_alpha)
        x = self.leaky_relu(self.groupNorms[5](self.conv_transpose_layers[5](x)), self.leaky_relu_alpha)
        x = self.leaky_relu(self.groupNorms[6](self.conv_transpose_layers[6](x)), self.leaky_relu_alpha)

        x = jax.image.resize(image=x, shape=(x.shape[0], x.shape[1] * 2, x.shape[2] * 2), method="bilinear")
        x = jnp.concatenate([x, skip_1], axis=0)

        x = self.leaky_relu(self.groupNorms[7](self.conv_transpose_layers[7](x)), self.leaky_relu_alpha)
        x = self.leaky_relu(self.groupNorms[8](self.conv_transpose_layers[8](x)), self.leaky_relu_alpha)
        x = self.leaky_relu(self.groupNorms[9](self.conv_transpose_layers[9](x)), self.leaky_relu_alpha)

        x = self.conv2d_output(x)
        return x


class LocNet(eqx.Module):
    Encoder_model: Encoder
    Decoder_model: Decoder

    def __init__(self,
                 enc_key,
                 dec_key,
                 enc_in=2,
                 enc_out=4,
                 dec_in=4,
                 dec_out=1,
                 n_dim=27,
                 leaky_relu_alpha=0.3,
                 ):
        self.Encoder_model = Encoder(
            enc_in=enc_in,
            enc_out=enc_out,
            n_dim=n_dim,
            leaky_relu_alpha=leaky_relu_alpha,
            key=enc_key
        )

        self.Decoder_model = Decoder(
            dec_in=dec_in,
            dec_out=dec_out,
            n_dim=n_dim,
            leaky_relu_alpha=leaky_relu_alpha,
            key=dec_key
        )

    def __call__(
            self,
            locNet_inputs: Float[Array, "2 256 256"],
    ):
        x, skip_1, skip_2, skip_3 = self.Encoder_model(locNet_inputs)
        x = self.Decoder_model(x, skip_1, skip_2, skip_3)
        return x


@eqx.filter_jit
def Focal_Loss(
        y: Float[Array, "batch 1 256 256"],
        pred_y: Float[Array, "batch 1 256 256"],
        gamma: Int[Array, ""],
        alpha: Float[Array, ""]
) -> Float[Array, ""]:
    target = y.squeeze()
    pred_y = pred_y.squeeze()
    bce = optax.sigmoid_binary_cross_entropy(logits=pred_y, labels=target)
    pred_y = jnp.exp(-1 * bce) * target + (1 - jnp.exp(-1 * bce)) * (1 - target)
    z = (1 - pred_y) * target
    z += (1 - target) * pred_y
    z = z ** gamma[0]
    scale = target * alpha[0]
    scale += (1 - target) * (1 - alpha[0])
    return jnp.sum(z * bce * scale)


@eqx.filter_jit
def loss(
        model: LocNet,
        x: Float[Array, "batch 3 32 32"],
        y: Int[Array, "batch"],
        gamma: Int[Array, ""],
        alpha: Float[Array, ""]
) -> Float[Array, ""]:
    pred_y = jax.vmap(model)(x)
    return Focal_Loss(y=y, pred_y=pred_y, gamma=gamma, alpha=alpha)


@eqx.filter_jit
def make_step(
        model: LocNet,
        optim: optax.GradientTransformation,
        opt_state: PyTree,
        x: Float[Array, "batch 3 32 32"],
        y: Int[Array, "batch"],
        gamma: Int[Array, ""],
        alpha: Float[Array, ""]
):
    loss_value, grads = eqx.filter_value_and_grad(loss)(model, x, y, gamma, alpha)
    updates, opt_state = optim.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss_value


@eqx.filter_jit
def compute_dist_loss(
        inference_model: LocNet,
        x: Float[Array, "batch 2 256 256"],
        y: Int[Array, "batch 1 256 256"]
) -> Float[Array, ""]:
    pred_y = jax.vmap(inference_model)(x)
    pred_y = jnp.reshape(jax.nn.sigmoid(pred_y).squeeze(),
                         (pred_y.shape[0], pred_y.shape[2] * pred_y.shape[3]))
    targets = jnp.reshape(y.squeeze(),
                          (y.shape[0], y.shape[2] * y.shape[3]))

    pred_argmax = jnp.argmax(pred_y, axis=1)
    target_argmax = jnp.argmax(targets, axis=1)

    targets_rows, targets_cols = target_argmax // 256, target_argmax % 256
    pred_rows, pred_cols = pred_argmax // 256, pred_argmax % 256

    euclidean_dist_no_reduction = jnp.sqrt((pred_rows - targets_rows) ** 2 + (pred_cols - targets_cols) ** 2)
    return euclidean_dist_no_reduction
    # return jnp.sum(euclidean_dist_no_reduction)


def evaluate(
        model: LocNet,
        testloader: DataLoader
) -> float:
    dist = 0
    inference_model = eqx.nn.inference_mode(model)
    for x, y in testloader:
        dist += jnp.sum(compute_dist_loss(inference_model=inference_model, x=x, y=y)).item()
    return dist


def test(
        model: LocNet,
        testloader: DataLoader
):
    from tqdm import tqdm
    dist = []
    # model_loaded = eqx.tree_deserialise_leaves(
    #     '/home/UNT/tdl0134/SOTA/eqx_SelfAttention_LocNet_0_001_90.eqx',
    #     model
    # )

    model_loaded = eqx.tree_deserialise_leaves(
        '/home/thanhle/SOTA/LocNet_GIN_K_20_dynamic_96_SOTA.eqx',
        model
    )

    inference_model = eqx.nn.inference_mode(
        model_loaded
    )
    for x, y in tqdm(testloader):
        dist.append(
            compute_dist_loss(
                inference_model=inference_model,
                x=x,
                y=y
            )
        )

    dist = np.array(dist)
    return np.sqrt(np.sum((dist ** 2)) / 40000)


def train(
        model: LocNet,
        trainloader: DataLoader,
        testloader: DataLoader,
        optim: optax.GradientTransformation,
        EPOCHS: int,
        GAMMA: Int[Array, ""],
        ALPHA: Float[Array, ""]
):
    opt_state = optim.init(eqx.filter(model, eqx.is_array))
    dist_loss = np.inf
    for epoch in range(EPOCHS):
        n_batches = len(trainloader)
        pBar = Progbar(target=n_batches, verbose=1)
        print(f"Starting epoch {epoch + 1}")

        for t, data in enumerate(trainloader):
            x, y = data
            model, opt_state, train_loss = make_step(model=model,
                                                     opt_state=opt_state,
                                                     optim=optim,
                                                     x=x,
                                                     y=y,
                                                     gamma=GAMMA,
                                                     alpha=ALPHA)

            pBar.update(t, values=[("loss", train_loss.item())])

        dist = evaluate(model=model,
                        testloader=testloader)

        pBar.update(n_batches, values=[("Dist sum", dist)])
        if dist < dist_loss:
            dist_loss = dist
            eqx.tree_serialise_leaves(
                f"/home/thanhle/SOTA/LocNet_GraphSage_dynamic_{epoch + 1}.eqx",
                model
            )


if __name__ == "__main__":
    import os

    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.90"
    os.environ["KERAS_BACKEND"] = "jax"
    from keras.utils import Progbar
    import torch

    SEED = 0
    lr = 5e-4
    BATCH = 64
    EPOCHS = 100
    GAMMA = jnp.array([3, ], dtype=jnp.int32)
    ALPHA = jnp.array([0.75, ], dtype=jnp.float32)
    key = jax.random.PRNGKey(SEED)

    random.seed(SEED)
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # Train = RadioMapDataset('/home/thanhle', 'Train_0_0001_TO_0_001')
    # Val = RadioMapDataset('/home/thanhle', 'Val_0_0001_TO_0_001')
    Test = RadioMapDataset('/home/thanhle', 'Test_data/Test_0_0001_TO_0_001')

    # Train_Loader = DataLoader(Train, shuffle=True, batch_size=BATCH, drop_last=True, collate_fn=collate_fn)
    # Val_Loader = DataLoader(Val, shuffle=True, batch_size=BATCH, collate_fn=collate_fn)
    Test_Loader = DataLoader(Test, shuffle=True, batch_size=BATCH, collate_fn=collate_fn)

    key, enc_key, dec_key = jax.random.split(key, 3)

    model = LocNet(enc_in=2,
                   enc_out=4,
                   dec_in=4,
                   dec_out=1,
                   n_dim=27,
                   leaky_relu_alpha=0.3,
                   enc_key=enc_key,
                   dec_key=dec_key)

    optimizer = optax.adamw(lr)

    result = test(
        model=model,
        testloader=Test_Loader
    )
    print(result)
    # train(
    #     model=model,
    #     trainloader=Train_Loader,
    #     testloader=Val_Loader,
    #     optim=optimizer,
    #     EPOCHS=EPOCHS,
    #     GAMMA=GAMMA,
    #     ALPHA=ALPHA
    # )
