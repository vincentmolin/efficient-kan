import jax
import jax.numpy as jnp
import jax.random as jr

import flax.linen as nn

from typing import List, Tuple


def b_splines(x: jnp.ndarray, grid: jnp.ndarray, spline_order: int):
    """
    Compute the B-spline bases for the given input array.

    Args:
        x: Input tensor of shape (batch_size, in_features).
        grid: Grid tensor of shape (in_features, grid_size + 2 * spline_order + 1).
    Returns:
        torch.Tensor: B-spline bases tensor of shape (batch_size, in_features, grid_size + spline_order).
    """
    x = jnp.expand_dims(x, -1)
    bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).astype(x.dtype)
    for k in range(1, spline_order + 1):
        bases = (
            (x - grid[:, : -(k + 1)])
            / (grid[:, k:-1] - grid[:, : -(k + 1)])
            * bases[:, :, :-1]
        ) + (
            (grid[:, k + 1 :] - x)
            / (grid[:, k + 1 :] - grid[:, 1:(-k)])
            * bases[:, :, 1:]
        )

    return bases


def curve2coeff(x: jnp.ndarray, y: jnp.ndarray, grid, spline_order):
    """
    Compute the coefficients of the curve that interpolates the given points.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, in_features).
        y (torch.Tensor): Output tensor of shape (batch_size, in_features, out_features).

    Returns:
        torch.Tensor: Coefficients tensor of shape (out_features, in_features, grid_size + spline_order).
    """
    assert len(x.shape) == 2
    assert len(y.shape) == 3
    # in_features = x.shape[-1]
    # out_features = y.shape[-1]

    A = b_splines(x, grid, spline_order).transpose(
        [1, 0, 2]
    )  # (in_features, batch_size, grid_size + spline_order)
    B = y.transpose([1, 0, 2])  # (in_features, batch_size, out_features)

    solution = jax.vmap(lambda a, b: jnp.linalg.lstsq(a, b)[0], in_axes=(0, 0))(
        A, B
    )  # (in_features, grid_size + spline_order, out_features)
    result = solution.transpose(
        [2, 0, 1]
    )  # (out_features, in_features, grid_size + spline_order)

    return result


class KANLinear(nn.Module):
    out_features: int
    grid_size: int = 5
    spline_order: int = 3
    scale_noise: float = 0.1
    scale_base: float = 1.0
    scale_spline: float = 1.0
    enable_standalone_scale_spline: bool = True
    grid_eps: float = 0.02
    grid_range: Tuple[int, ...] = (-1, 1)

    @nn.compact
    def __call__(self, x, update_grid=False):
        assert len(x.shape) == 2
        in_features = x.shape[1]

        grid = self.variable("batch_stats", "grid", self._grid_init, in_features)

        base_weight = self.param(
            "base_weight",
            nn.initializers.lecun_normal(),
            (x.shape[-1], self.out_features),
        )

        spline_weight = self.variable(
            "params",
            "spline_weight",
            lambda: self._spline_init(self.make_rng(), in_features, grid.value),
        )

        if self.enable_standalone_scale_spline:
            spline_scaler = self.param(
                "spline_scaler",
                nn.initializers.lecun_normal(),
                (self.out_features, in_features),
            )

            scaled_spline_weight = spline_weight.value * spline_scaler[..., None]
        else:
            scaled_spline_weight = spline_weight.value

        if update_grid:
            new_grid, new_spline_weights = self._update_grid(
                grid.value, x, scaled_spline_weight
            )
            grid.value = new_grid
            spline_weight.value = new_spline_weights

        base_output = (
            jax.nn.silu(x)
            @ base_weight
            # self.base_activation(x) @ base_weight
        )  # F.linear(self.base_activation(x), self.base_weight)
        b_spline_out = b_splines(x, grid.value, self.spline_order).reshape(
            (x.shape[0], -1)
        )
        spline_output = b_spline_out @ scaled_spline_weight.reshape(
            (-1, self.out_features)
        )
        return base_output + spline_output

    def _spline_init(self, rng, in_features, grid):
        noise = jr.uniform(rng, (self.grid_size + 1, in_features, self.out_features))
        noise = (noise - 1 / 2) * self.scale_noise / self.grid_size
        return curve2coeff(
            grid.T[self.spline_order : -self.spline_order],
            noise,
            grid,
            self.spline_order,
        ) * (self.scale_spline if not self.enable_standalone_scale_spline else 1.0)

    def _grid_init(self, in_features):
        h = (self.grid_range[1] - self.grid_range[0]) / self.grid_size
        grid = (
            jnp.arange(-self.spline_order, self.grid_size + self.spline_order + 1) * h
            + self.grid_range[0]
        )
        grid = jnp.repeat(grid.reshape(1, -1), in_features, 0)
        return grid

    def _update_grid(self, grid, x, scaled_spline_weight, margin=0.01):
        batch_size = x.shape[0]

        splines = b_splines(x, grid, self.spline_order)  # (batch, in, coeff)
        splines = splines.transpose((1, 0, 2))  # (in, batch, coeff)
        orig_coeff = scaled_spline_weight.transpose((1, 2, 0))  # (out, in, coeff)
        unreduced_spline_output = jnp.matmul(splines, orig_coeff)  # (in, batch, out)
        unreduced_spline_output = unreduced_spline_output.transpose((1, 0, 2))
        #     # sort each channel individually to collect data distribution
        x_sorted = jnp.sort(x, axis=0)
        grid_adaptive = x_sorted[
            jnp.linspace(0, batch_size - 1, self.grid_size + 1, dtype=jnp.int32)
        ]

        uniform_step = (x_sorted[-1] - x_sorted[0] + 2 * margin) / self.grid_size
        grid_uniform = (
            jnp.expand_dims(jnp.arange(self.grid_size + 1, dtype=jnp.float32), 1)
            * uniform_step
            + x_sorted[0]
            - margin
        )

        new_grid = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive
        new_grid = jnp.concatenate(
            [
                new_grid[:1]
                - uniform_step
                * jnp.expand_dims(jnp.arange(self.spline_order, 0, -1), 1),
                new_grid,
                new_grid[-1:]
                + uniform_step
                * jnp.expand_dims(jnp.arange(1, self.spline_order + 1), 1),
            ],
            axis=0,
        ).T

        spline_weight = curve2coeff(
            x, unreduced_spline_output, new_grid, self.spline_order
        )

        return new_grid, spline_weight

    # def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
    #     """
    #     Compute the regularization loss.

    #     This is a dumb simulation of the original L1 regularization as stated in the
    #     paper, since the original one requires computing absolutes and entropy from the
    #     expanded (batch, in_features, out_features) intermediate tensor, which is hidden
    #     behind the F.linear function if we want an memory efficient implementation.

    #     The L1 regularization is now computed as mean absolute value of the spline
    #     weights. The authors implementation also includes this term in addition to the
    #     sample-based regularization.
    #     """
    #     l1_fake = self.spline_weight.abs().mean(-1)
    #     regularization_loss_activation = l1_fake.sum()
    #     p = l1_fake / regularization_loss_activation
    #     regularization_loss_entropy = -torch.sum(p * p.log())
    #     return (
    #         regularize_activation * regularization_loss_activation
    #         + regularize_entropy * regularization_loss_entropy
    #     )


class KAN(nn.Module):
    features: Tuple[int, ...] = (4, 4)
    grid_size: int = 5
    spline_order: int = 3
    scale_noise: float = 0.1
    scale_base: float = 1.0
    scale_spline: float = 1.0
    enable_standalone_scale_spline: bool = True
    grid_eps: float = 0.02
    grid_range: Tuple[int, ...] = (-1, 1)

    @nn.compact
    def __call__(self, x, update_grid=False):
        for i, f in enumerate(self.features):
            x = KANLinear(
                f,
                grid_size=self.grid_size,
                spline_order=self.spline_order,
                scale_noise=self.scale_noise,
                scale_base=self.scale_base,
                scale_spline=self.scale_spline,
                grid_eps=self.grid_eps,
                grid_range=self.grid_range,
            )(x, update_grid=update_grid)
        return x

    # def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
    #     return sum(
    #         layer.regularization_loss(regularize_activation, regularize_entropy)
    #         for layer in self.layers
    #     )


if __name__ == "__main__":
    import jax
    import jax.numpy as jnp
    import jax.random as jr

    import optax

    from tqdm import tqdm

    kan = KAN([4, 1])

    x = jnp.linspace(-1, 1, 1024).reshape(-1, 1)
    y = jnp.sin(2 * jnp.pi * x) + 0.051 * jr.normal(jr.key(1), x.shape)

    vars = kan.init(jr.key(0), x)

    print(f"number of parameters: {sum(v.size for v in jax.tree.flatten(vars)[0])}")

    tx = optax.adam(1e-2)

    params = vars["params"]
    state = vars["batch_stats"]
    opt_state = tx.init(params)

    def update_step(params, state, opt_state, x, y):
        def loss_fn(params):
            y_hat = kan.apply({"params": params, "batch_stats": state}, x)
            return jnp.mean((y - y_hat) ** 2)

        loss, grad = jax.value_and_grad(loss_fn)(params)
        updates, opt_state = tx.update(grad, opt_state)
        new_params = optax.apply_updates(params, updates)
        return new_params, opt_state, loss

    for i in tqdm(range(100)):
        params, opt_state, loss = update_step(params, state, opt_state, x, y)
        print(loss)

    import matplotlib.pyplot as plt

    x_test = jnp.linspace(-1.5, 1.5, 1024).reshape(-1, 1)
    y_test = kan.apply({"params": params, "batch_stats": state}, x_test)
    plt.plot(x_test, y_test)
    plt.scatter(x, y)
    # pbar.set_postfix(mse_loss=loss.item(), reg_loss=reg_loss.item())
    # for layer in kan.layers:
    #     print(layer.spline_weight)
