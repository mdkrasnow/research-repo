def none_or_str(value):
    if value == 'None':
        return None
    return value

def parse_transport_args(parser):
    group = parser.add_argument_group("Transport arguments")
    group.add_argument("--path-type", type=str, default="Linear", choices=["Linear", "GVP", "VP"])
    group.add_argument("--prediction", type=str, default="velocity", choices=["velocity", "score", "noise"])
    group.add_argument("--loss-weight", type=none_or_str, default=None, choices=[None, "velocity", "likelihood"])
    group.add_argument("--sample-eps", type=float)
    group.add_argument("--train-eps", type=float)
    group.add_argument("--corruption-mode", type=str, default="gaussian",
        choices=["gaussian", "mask", "fourier", "blur", "downsample", "mixture"],
        help="x0 start-state family: Gaussian (baseline), Bernoulli mask, Fourier low-pass, Gaussian blur, downsample/upsample, or weighted mixture")
    group.add_argument("--mask-prob", type=float, default=0.5,
        help="fraction of latent elements replaced by noise under mask corruption")
    group.add_argument("--fourier-cutoff", type=float, default=0.25,
        help="fraction of radial frequency band kept under Fourier corruption")
    group.add_argument("--blur-sigma", type=float, default=1.0,
        help="Gaussian blur kernel sigma (latent-space pixels) under blur corruption")
    group.add_argument("--downsample-factor", type=float, default=4.0,
        help="spatial downsample/upsample factor under downsample corruption")
    group.add_argument("--gaussian-weight", type=float, default=1.0,
        help="mixture weight lambda_G (only used when --corruption-mode mixture)")
    group.add_argument("--mask-weight", type=float, default=0.0,
        help="mixture weight lambda_M (only used when --corruption-mode mixture)")
    group.add_argument("--fourier-weight", type=float, default=0.0,
        help="mixture weight lambda_F (only used when --corruption-mode mixture)")
    group.add_argument("--blur-weight", type=float, default=0.0,
        help="mixture weight lambda_B (only used when --corruption-mode mixture)")
    group.add_argument("--downsample-weight", type=float, default=0.0,
        help="mixture weight lambda_D (only used when --corruption-mode mixture)")


def parse_ode_args(parser):
    group = parser.add_argument_group("ODE arguments")
    group.add_argument("--sampling-method", type=str, default="dopri5", help="blackbox ODE solver methods; for full list check https://github.com/rtqichen/torchdiffeq")
    group.add_argument("--atol", type=float, default=1e-6, help="Absolute tolerance")
    group.add_argument("--rtol", type=float, default=1e-3, help="Relative tolerance")
    group.add_argument("--reverse", action="store_true")
    group.add_argument("--likelihood", action="store_true")

def parse_sde_args(parser):
    group = parser.add_argument_group("SDE arguments")
    group.add_argument("--sampling-method", type=str, default="Euler", choices=["Euler", "Heun"])
    group.add_argument("--diffusion-form", type=str, default="sigma", \
                        choices=["constant", "SBDM", "sigma", "linear", "decreasing", "increasing-decreasing"],\
                        help="form of diffusion coefficient in the SDE")
    group.add_argument("--diffusion-norm", type=float, default=1.0)
    group.add_argument("--last-step", type=none_or_str, default="Mean", choices=[None, "Mean", "Tweedie", "Euler"],\
                        help="form of last step taken in the SDE")
    group.add_argument("--last-step-size", type=float, default=0.04, \
                        help="size of the last step taken")