import torch as th
import numpy as np
import logging
import torch.distributed as dist

import enum

from . import path
from .utils import EasyDict, log_state, mean_flat
from .integrators import ode, sde

class ModelType(enum.Enum):
    """
    Which type of output the model predicts.
    """

    NOISE = enum.auto()  # the model predicts epsilon
    SCORE = enum.auto()  # the model predicts \nabla \log p(x)
    VELOCITY = enum.auto()  # the model predicts v(x)

class PathType(enum.Enum):
    """
    Which type of path to use.
    """

    LINEAR = enum.auto()
    GVP = enum.auto()
    VP = enum.auto()

class WeightType(enum.Enum):
    """
    Which type of weighting to use.
    """

    NONE = enum.auto()
    VELOCITY = enum.auto()
    LIKELIHOOD = enum.auto()


class Transport:

    def __init__(
        self,
        *,
        model_type,
        path_type,
        loss_type,
        train_eps,
        sample_eps,
    ):
        path_options = {
            PathType.LINEAR: path.ICPlan,
            PathType.GVP: path.GVPCPlan,
            PathType.VP: path.VPCPlan,
        }

        self.loss_type = loss_type
        self.model_type = model_type
        self.path_sampler = path_options[path_type]()
        self.train_eps = train_eps
        self.sample_eps = sample_eps

    def prior_logp(self, z):
        '''
            Standard multivariate normal prior
            Assume z is batched
        '''
        shape = th.tensor(z.size())
        N = th.prod(shape[1:])
        _fn = lambda x: -N / 2. * np.log(2 * np.pi) - th.sum(x ** 2) / 2.
        return th.vmap(_fn)(z)
    

    def check_interval(
        self, 
        train_eps, 
        sample_eps, 
        *, 
        diffusion_form="SBDM",
        sde=False, 
        reverse=False, 
        eval=False,
        last_step_size=0.0,
    ):
        t0 = 0
        t1 = 1
        eps = train_eps if not eval else sample_eps
        if (type(self.path_sampler) in [path.VPCPlan]):

            t1 = 1 - eps if (not sde or last_step_size == 0) else 1 - last_step_size

        elif (type(self.path_sampler) in [path.ICPlan, path.GVPCPlan]) \
            and (self.model_type != ModelType.VELOCITY or sde): # avoid numerical issue by taking a first semi-implicit step

            t0 = eps if (diffusion_form == "SBDM" and sde) or self.model_type != ModelType.VELOCITY else 0
            t1 = 1 - eps if (not sde or last_step_size == 0) else 1 - last_step_size
        
        if reverse:
            t0, t1 = 1 - t0, 1 - t1

        return t0, t1


    def sample(self, x1):
        """Sampling x0 & t based on shape of x1 (if needed)
          Args:
            x1 - data point; [batch, *dim]
        """
        
        x0 = th.randn_like(x1)
        t0, t1 = self.check_interval(self.train_eps, self.sample_eps)
        t = th.rand((x1.shape[0],)) * (t1 - t0) + t0
        t = t.to(x1)
        return t, x0, x1

    def disp_loss(self, z): # Dispersive Loss implementation (InfoNCE-L2 variant)
        z = z.reshape((z.shape[0],-1)) # flatten
        diff = th.nn.functional.pdist(z).pow(2)/z.shape[1] # normalize by dimension
        diff = th.concat((diff, diff, th.zeros(z.shape[0]).cuda()))  # match JAX implementation of full BxB matrix
        return th.log(th.exp(-diff).mean())

    def get_ct(self, t): #ct implementation
        interp = 0.8
        start = 1.0
        ct = th.minimum(start-(start-1)/(interp)*t, 1/(1-interp)-1/(1-interp)*t)*4
        return ct

    def mine_negative(self, model, t, x0, x1, model_kwargs, anm):
        """Adversarial Negative Mining (ANM) -- loss-ascent / PGD variant.

        EqM's negative is the Gaussian endpoint x0 ~ N(0, I) (see sample()).
        Here we replace it with a *hard* endpoint x0_adv that locally MAXIMIZES
        the EqM regression residual ||f(x_t) - u_t||^2 at the sampled t, via a
        few projected-gradient-ASCENT steps on the endpoint. This is the
        AT-EBM / CEM adversarial-negative pattern ("PGD attack = non-convergent
        sampler") specialized to EqM's regression target, and the v10-style
        hard-example idea: train where the current field is worst.

        Key design choices (vs. mirroring the inference sampler):
        - We ascend the LOSS w.r.t. the endpoint, not follow the generation
          field. The inference step size (0.0017, tuned for 250 steps) is a
          no-op over a few steps; loss-ascent moves the endpoint meaningfully.
        - Step size and trust region are RELATIVE to per-sample ||x0||, so the
          mining magnitude is dimension-free.
        - Params are held fixed (grad taken only w.r.t. the endpoint) and the
          result is stop-gradient'd by the caller, so the outer EqM target
          geometry (x1 - x0_adv) * c(t) is preserved exactly.

        Returns (x0_adv, diagnostics).
        """
        net = model.module if hasattr(model, "module") else model  # bypass DDP reducer
        is_ebm = getattr(net, "ebm", "none") != "none"
        # forward keys label-dropout on self.training, and uses create_graph=train
        # for the EqM-E inner autograd.grad; need that graph only in ebm modes.
        mk = {**model_kwargs, "return_act": False, "train": is_ebm}

        steps = int(anm["steps"])
        rel_step = float(anm["step_size"])          # per-step displacement as fraction of ||x0||
        rel_ball = float(anm.get("eps_ball", 0.0))  # L2 trust-region radius as fraction of ||x0|| (0=off)

        def _flat_norm(z):
            return z.flatten(1).norm(dim=1).view(-1, *([1] * (z.ndim - 1)))

        x0_init = x0.detach()
        norm0 = _flat_norm(x0_init).clamp_min(1e-12)
        ct = self.get_ct(t)[:, None, None, None]

        def _residual(endpoint):
            # Same path + target construction as training_losses, at the sampled t.
            _, xt, ut = self.path_sampler.plan(t, endpoint, x1)
            ut = ut * ct
            out = net(xt, t, **mk)
            if not th.is_tensor(out):  # forward may return (field, act)/(field, -E)
                out = out[0]
            return mean_flat((out - ut) ** 2)

        was_training = net.training
        net.eval()  # disable stochastic label dropout so the mined negative is deterministic
        try:
            with th.enable_grad():  # residual eval needs grad in ebm modes (field via autograd.grad)
                res_before = _residual(x0_init).detach()

            delta = th.zeros_like(x0_init)
            for _ in range(steps):
                endpoint = (x0_init + delta).detach().requires_grad_(True)
                with th.enable_grad():
                    res = _residual(endpoint)
                    g = th.autograd.grad(res.sum(), endpoint, create_graph=False)[0]
                g = g / _flat_norm(g).clamp_min(1e-12)        # per-sample unit ascent direction
                delta = delta + rel_step * norm0 * g          # ascend the residual
                if rel_ball > 0:                              # project to L2 trust region
                    dnorm = _flat_norm(delta).clamp_min(1e-12)
                    delta = delta * (rel_ball * norm0 / dnorm).clamp(max=1.0)
                delta = delta.detach()

            x0_adv = (x0_init + delta).detach()
            with th.enable_grad():
                res_after = _residual(x0_adv).detach()
            disp_ratio = (_flat_norm(x0_adv - x0_init) / norm0).mean()
        finally:
            net.train(was_training)

        diag = {
            "anm_disp_ratio": disp_ratio.detach(),     # ||x0_adv - x0|| / ||x0||; ~0 means mining is a no-op
            "anm_res_before": res_before.mean().detach(),
            "anm_res_after": res_after.mean().detach(),
        }
        return x0_adv, diag

    def training_losses(
        self,
        model,
        x1,
        model_kwargs=None,
        anm=None,
    ):
        """Loss for training the score model
        Args:
        - model: backbone model; could be score, noise, or velocity
        - x1: datapoint
        - model_kwargs: additional arguments for the model
        """
        if model_kwargs == None: 
            model_kwargs = {}
        
        t, x0, x1 = self.sample(x1)

        # Adversarial Negative Mining: replace the random Gaussian endpoint x0
        # with a hard endpoint that locally maximizes the EqM residual (PGD
        # ascent). stop-grad keeps the EqM target geometry (x1 - x0)*c(t) exact.
        anm_diag = {}
        if anm is not None and anm.get("steps", 0) > 0:
            x0, anm_diag = self.mine_negative(model, t, x0, x1, model_kwargs, anm)

        t, xt, ut = self.path_sampler.plan(t, x0, x1)
        ut = ut * self.get_ct(t)[:,None,None,None] # use energy-compatible target
        model_output = model(xt, t, **model_kwargs)
        disp_loss = 0

        # get intermediate activation and apply Dispersive Loss
        if "return_act" in model_kwargs and model_kwargs['return_act']:
            model_output, act = model_output
            disp_loss = self.disp_loss(act[len(act)-1])
        
        B, *_, C = xt.shape
        assert model_output.size() == (B, *xt.size()[1:-1], C)

        terms = {}
        terms['pred'] = model_output
        if self.model_type == ModelType.VELOCITY:
            terms['loss'] = mean_flat(((model_output - ut) ** 2))
        else: 
            _, drift_var = self.path_sampler.compute_drift(xt, t)
            sigma_t, _ = self.path_sampler.compute_sigma_t(path.expand_t_like_x(t, xt))
            if self.loss_type in [WeightType.VELOCITY]:
                weight = (drift_var / sigma_t) ** 2
            elif self.loss_type in [WeightType.LIKELIHOOD]:
                weight = drift_var / (sigma_t ** 2)
            elif self.loss_type in [WeightType.NONE]:
                weight = 1
            else:
                raise NotImplementedError()
            
            if self.model_type == ModelType.NOISE:
                terms['loss'] = mean_flat(weight * ((model_output - x0) ** 2))
            else:
                terms['loss'] = mean_flat(weight * ((model_output * sigma_t + x0) ** 2))
        terms['loss'] += 0.5*disp_loss
        terms.update(anm_diag)  # ANM diagnostics (empty dict when ANM disabled)
        return terms
    

    def get_drift(
        self
    ):
        """member function for obtaining the drift of the probability flow ODE"""
        def score_ode(x, t, model, **model_kwargs):
            drift_mean, drift_var = self.path_sampler.compute_drift(x, t)
            model_output = model(x, t, **model_kwargs)
            return (-drift_mean + drift_var * model_output) # by change of variable
        
        def noise_ode(x, t, model, **model_kwargs):
            drift_mean, drift_var = self.path_sampler.compute_drift(x, t)
            sigma_t, _ = self.path_sampler.compute_sigma_t(path.expand_t_like_x(t, x))
            model_output = model(x, t, **model_kwargs)
            score = model_output / -sigma_t
            return (-drift_mean + drift_var * score)
        
        def velocity_ode(x, t, model, **model_kwargs):
            model_output = model(x, t, **model_kwargs)
            return model_output

        if self.model_type == ModelType.NOISE:
            drift_fn = noise_ode
        elif self.model_type == ModelType.SCORE:
            drift_fn = score_ode
        else:
            drift_fn = velocity_ode
        
        def body_fn(x, t, model, **model_kwargs):
            model_output = drift_fn(x, t, model, **model_kwargs)
            assert model_output.shape == x.shape, "Output shape from ODE solver must match input shape"
            return model_output

        return body_fn
    

    def get_score(
        self,
    ):
        """member function for obtaining score of 
            x_t = alpha_t * x + sigma_t * eps"""
        if self.model_type == ModelType.NOISE:
            score_fn = lambda x, t, model, **kwargs: model(x, t, **kwargs) / -self.path_sampler.compute_sigma_t(path.expand_t_like_x(t, x))[0]
        elif self.model_type == ModelType.SCORE:
            score_fn = lambda x, t, model, **kwagrs: model(x, t, **kwagrs)
        elif self.model_type == ModelType.VELOCITY:
            score_fn = lambda x, t, model, **kwargs: self.path_sampler.get_score_from_velocity(model(x, t, **kwargs), x, t)
        else:
            raise NotImplementedError()
        
        return score_fn


class Sampler:
    """Sampler class for the transport model"""
    def __init__(
        self,
        transport,
    ):
        """Constructor for a general sampler; supporting different sampling methods
        Args:
        - transport: an tranport object specify model prediction & interpolant type
        """
        
        self.transport = transport
        self.drift = self.transport.get_drift()
        self.score = self.transport.get_score()
    
    def __get_sde_diffusion_and_drift(
        self,
        *,
        diffusion_form="SBDM",
        diffusion_norm=1.0,
    ):

        def diffusion_fn(x, t):
            diffusion = self.transport.path_sampler.compute_diffusion(x, t, form=diffusion_form, norm=diffusion_norm)
            return diffusion
        
        sde_drift = \
            lambda x, t, model, **kwargs: \
                self.drift(x, t, model, **kwargs) + diffusion_fn(x, t) * self.score(x, t, model, **kwargs)
    
        sde_diffusion = diffusion_fn

        return sde_drift, sde_diffusion
    
    def __get_last_step(
        self,
        sde_drift,
        *,
        last_step,
        last_step_size,
    ):
        """Get the last step function of the SDE solver"""
    
        if last_step is None:
            last_step_fn = \
                lambda x, t, model, **model_kwargs: \
                    x
        elif last_step == "Mean":
            last_step_fn = \
                lambda x, t, model, **model_kwargs: \
                    x + sde_drift(x, t, model, **model_kwargs) * last_step_size
        elif last_step == "Tweedie":
            alpha = self.transport.path_sampler.compute_alpha_t # simple aliasing; the original name was too long
            sigma = self.transport.path_sampler.compute_sigma_t
            last_step_fn = \
                lambda x, t, model, **model_kwargs: \
                    x / alpha(t)[0][0] + (sigma(t)[0][0] ** 2) / alpha(t)[0][0] * self.score(x, t, model, **model_kwargs)
        elif last_step == "Euler":
            last_step_fn = \
                lambda x, t, model, **model_kwargs: \
                    x + self.drift(x, t, model, **model_kwargs) * last_step_size
        else:
            raise NotImplementedError()

        return last_step_fn

    def sample_sde(
        self,
        *,
        sampling_method="Euler",
        diffusion_form="SBDM",
        diffusion_norm=1.0,
        last_step="Mean",
        last_step_size=0.04,
        num_steps=250,
    ):
        """returns a sampling function with given SDE settings
        Args:
        - sampling_method: type of sampler used in solving the SDE; default to be Euler-Maruyama
        - diffusion_form: function form of diffusion coefficient; default to be matching SBDM
        - diffusion_norm: function magnitude of diffusion coefficient; default to 1
        - last_step: type of the last step; default to identity
        - last_step_size: size of the last step; default to match the stride of 250 steps over [0,1]
        - num_steps: total integration step of SDE
        """

        if last_step is None:
            last_step_size = 0.0

        sde_drift, sde_diffusion = self.__get_sde_diffusion_and_drift(
            diffusion_form=diffusion_form,
            diffusion_norm=diffusion_norm,
        )

        t0, t1 = self.transport.check_interval(
            self.transport.train_eps,
            self.transport.sample_eps,
            diffusion_form=diffusion_form,
            sde=True,
            eval=True,
            reverse=False,
            last_step_size=last_step_size,
        )

        _sde = sde(
            sde_drift,
            sde_diffusion,
            t0=t0,
            t1=t1,
            num_steps=num_steps,
            sampler_type=sampling_method
        )

        last_step_fn = self.__get_last_step(sde_drift, last_step=last_step, last_step_size=last_step_size)
            

        def _sample(init, model, **model_kwargs):
            xs = _sde.sample(init, model, **model_kwargs)
            ts = th.ones(init.size(0), device=init.device) * t1
            x = last_step_fn(xs[-1], ts, model, **model_kwargs)
            xs.append(x)

            assert len(xs) == num_steps, "Samples does not match the number of steps"

            return xs

        return _sample
    
    def sample_ode(
        self,
        *,
        sampling_method="dopri5",
        num_steps=50,
        atol=1e-6,
        rtol=1e-3,
        reverse=False,
    ):
        """returns a sampling function with given ODE settings
        Args:
        - sampling_method: type of sampler used in solving the ODE; default to be Dopri5
        - num_steps: 
            - fixed solver (Euler, Heun): the actual number of integration steps performed
            - adaptive solver (Dopri5): the number of datapoints saved during integration; produced by interpolation
        - atol: absolute error tolerance for the solver
        - rtol: relative error tolerance for the solver
        - reverse: whether solving the ODE in reverse (data to noise); default to False
        """
        if reverse:
            drift = lambda x, t, model, **kwargs: self.drift(x, th.ones_like(t) * (1 - t), model, **kwargs)
        else:
            drift = self.drift

        t0, t1 = self.transport.check_interval(
            self.transport.train_eps,
            self.transport.sample_eps,
            sde=False,
            eval=True,
            reverse=reverse,
            last_step_size=0.0,
        )

        _ode = ode(
            drift=drift,
            t0=t0,
            t1=t1,
            sampler_type=sampling_method,
            num_steps=num_steps,
            atol=atol,
            rtol=rtol,
        )
        
        return _ode.sample

    def sample_ode_likelihood(
        self,
        *,
        sampling_method="dopri5",
        num_steps=50,
        atol=1e-6,
        rtol=1e-3,
    ):
        
        """returns a sampling function for calculating likelihood with given ODE settings
        Args:
        - sampling_method: type of sampler used in solving the ODE; default to be Dopri5
        - num_steps: 
            - fixed solver (Euler, Heun): the actual number of integration steps performed
            - adaptive solver (Dopri5): the number of datapoints saved during integration; produced by interpolation
        - atol: absolute error tolerance for the solver
        - rtol: relative error tolerance for the solver
        """
        def _likelihood_drift(x, t, model, **model_kwargs):
            x, _ = x
            eps = th.randint(2, x.size(), dtype=th.float, device=x.device) * 2 - 1
            t = th.ones_like(t) * (1 - t)
            with th.enable_grad():
                x.requires_grad = True
                grad = th.autograd.grad(th.sum(self.drift(x, t, model, **model_kwargs) * eps), x)[0]
                logp_grad = th.sum(grad * eps, dim=tuple(range(1, len(x.size()))))
                drift = self.drift(x, t, model, **model_kwargs)
            return (-drift, logp_grad)
        
        t0, t1 = self.transport.check_interval(
            self.transport.train_eps,
            self.transport.sample_eps,
            sde=False,
            eval=True,
            reverse=False,
            last_step_size=0.0,
        )

        _ode = ode(
            drift=_likelihood_drift,
            t0=t0,
            t1=t1,
            sampler_type=sampling_method,
            num_steps=num_steps,
            atol=atol,
            rtol=rtol,
        )

        def _sample_fn(x, model, **model_kwargs):
            init_logp = th.zeros(x.size(0)).to(x)
            input = (x, init_logp)
            drift, delta_logp = _ode.sample(input, model, **model_kwargs)
            drift, delta_logp = drift[-1], delta_logp[-1]
            prior_logp = self.transport.prior_logp(drift)
            logp = prior_logp - delta_logp
            return logp, drift

        return _sample_fn