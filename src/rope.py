import torch


def compute_rope_params(
    head_dim,
    theta_base=10000.0,
    context_length=4096,
    dtype=torch.float32,
    device=None
):
    """
    Precalcula los valores coseno y seno necesarios para RoPE.

    Parámetros
    ----------
    head_dim : int
        Dimensión de cada cabeza de atención.
    theta_base : float
        Base utilizada para calcular las frecuencias.
    context_length : int
        Número máximo de posiciones.
    dtype : torch.dtype
        Tipo de dato de los tensores.
    device : str o torch.device
        CPU o GPU.

    Devuelve
    --------
    cos, sin : torch.Tensor
        Tensores de forma (context_length, head_dim)
    """

    assert head_dim % 2 == 0, "head_dim debe ser par"

    # Frecuencias inversas para cada par de dimensiones
    inv_freq = 1.0 / (
        theta_base ** (
            torch.arange(
                0,
                head_dim,
                2,
                dtype=dtype,
                device=device
            ) / head_dim
        )
    )

    # Posiciones: 0, 1, 2, ..., context_length-1
    positions = torch.arange(
        context_length,
        dtype=dtype,
        device=device
    )

    # Ángulo correspondiente a cada posición y frecuencia
    angles = (
        positions.unsqueeze(1)
        * inv_freq.unsqueeze(0)
    )

    # Duplicamos para cubrir toda head_dim
    angles = torch.cat(
        [angles, angles],
        dim=1
    )

    cos = torch.cos(angles)
    sin = torch.sin(angles)

    return cos, sin


def apply_rope(x, cos, sin):
    """
    Aplica Rotary Positional Embeddings a un tensor.

    x debe tener forma:
    (batch_size, num_heads, seq_len, head_dim)
    """

    _, _, seq_len, head_dim = x.shape

    assert head_dim % 2 == 0, "head_dim debe ser par"

    # Separamos el vector en dos mitades
    x1 = x[..., :head_dim // 2]
    x2 = x[..., head_dim // 2:]

    # Rotación de 90 grados:
    # [x1 | x2] -> [-x2 | x1]
    rotated = torch.cat(
        [-x2, x1],
        dim=-1
    )

    # Nos quedamos con las posiciones usadas
    # y añadimos dimensiones para batch y heads
    cos = (
        cos[:seq_len]
        .unsqueeze(0)
        .unsqueeze(0)
        .to(device=x.device, dtype=x.dtype)
    )

    sin = (
        sin[:seq_len]
        .unsqueeze(0)
        .unsqueeze(0)
        .to(device=x.device, dtype=x.dtype)
    )

    # Aplicamos la rotación
    x_rotated = (
        x * cos
        + rotated * sin
    )

    return x_rotated