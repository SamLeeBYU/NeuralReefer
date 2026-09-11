"""
Defines the per-submodel CNN classifier architecture, and the ensemble-
combination optimizers (EM, linear stacking, Adam, class-reweighting,
multinomial regression, and a small feedforward net) that combine the M
submodels' predictions into one final class distribution.
"""

import numpy as np
import torch
import torch.nn as nn
from torchvision.models import (
    resnet18, ResNet18_Weights,
    resnet34, ResNet34_Weights,
    resnet50, ResNet50_Weights,
    resnet101, ResNet101_Weights,
    resnet152, ResNet152_Weights
)

import torch.nn.functional as F

model_dict = {18: resnet18, 34: resnet34, 50: resnet50, 101: resnet101, 152: resnet152}
weights_dict = {18: ResNet18_Weights, 34: ResNet34_Weights, 50: ResNet50_Weights, 101: ResNet101_Weights, 152: ResNet152_Weights}

class CoralClassifier(nn.Module):

    # This module implements the CoralClassifier class, a deep convolutional neural network (CNN) optimized for
    # binary classification of segmented coral reef image crops. The model leverages a pretrained ResNet architecture,
    # replacing the final fully connected (FC) layer with a custom multi-layer perceptron (MLP) head composed of
    # ReLU activations and dropout regularization for improved generalization.

    # The last layer may be modified depending on the dimension of the output using the argument 'dim'

    def __init__(self, pretrained=True, dim=1, res=18):

        super(CoralClassifier, self).__init__()

        try:
            model = model_dict[res]
            weights = weights_dict[res]
        except KeyError:
            raise ValueError("Unsupported ResNet depth.")

        self.backbone = model(weights=weights.DEFAULT if pretrained else None)
        self.backbone.fc = self._create_fc(self.backbone.fc.in_features, dim=dim)

    @staticmethod
    def _create_fc(in_feautres, dim=1):

        return nn.Sequential(

            nn.Linear(in_feautres, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.7),

            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.6),

            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),

            nn.Linear(128, dim)
        )

    def forward(self, x):
        return self.backbone(x)

class EMEnsembleOptimizer:
    """
    Weighted-aggregation ensemble combiner, fit via Expectation-Maximization
    with an exact Newton/IRLS update for the per-submodel multinomial-logit
    calibration vectors. Pure NumPy -- no torch, no autodiff, no Adam.

    Model (see the multinomial-logit EM derivation this implements):
        p_i^(m)  = softmax(logit_i^(m))                                 -- fixed, pretrained submodel output
        u_i^(m)  = log p_i^(m)                                          -- fixed, per-submodel log-probabilities
        beta_m in R^K                                                    -- per-submodel, per-class calibration (unconstrained)
        tilde_p_{i,k}^(m)(beta_m) = softmax_k(beta_{m,k} u_{i,k}^(m))    -- Eq. 3 (vector-scaling calibration)
        p_hat_i = sum_m alpha_m tilde_p_i^(m)(beta_m),   alpha in simplex -- Eq. 2

    The M submodels are already trained and frozen by the time this runs --
    their logits are just fixed input data for the EM fit. The alpha-update
    is closed form; the beta_m-update is an exactly solvable *concave*
    weighted multinomial-logistic-regression problem (class k's "feature"
    is submodel m's own log-probability u_{i,k}^(m)), solved to global
    optimality every M-step by Newton's method. There's no local-optimum
    risk within a single submodel's calibration -- only the outer (alpha,
    {beta_m}) coupling remains non-convex, which is why fit() still uses
    multiple random restarts.
    """

    def __init__(self, M, K):
        self.M = M
        self.K = K
        self.alpha = np.full(M, 1.0 / M)
        self.beta = np.ones((M, K))  # beta_m = 1 (all-ones) reproduces each submodel's own p_i^(m) unchanged

    @staticmethod
    def _softmax(logits):
        z = logits - logits.max(axis=-1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=-1, keepdims=True)

    @staticmethod
    def _log_softmax(logits):
        z = logits - logits.max(axis=-1, keepdims=True)
        return z - np.log(np.exp(z).sum(axis=-1, keepdims=True))

    def _recalibrated_probs(self, u, beta=None):
        """Eq. 3. u: [N, M, K] (per-submodel log-probs) -> tilde_p: [N, M, K]."""
        beta = self.beta if beta is None else beta
        z = beta[None, :, :] * u                              # [N, M, K]
        return self._softmax(z)

    def predict_proba(self, logits):
        """logits: [N, M, K] raw submodel logits -> p_hat: [N, K] (Eq. 2)."""
        u = self._log_softmax(logits)
        tilde_p = self._recalibrated_probs(u)
        return (self.alpha[None, :, None] * tilde_p).sum(axis=1)

    def _fit_beta_m(self, u_m, y_idx, c_m, beta0, newton_iters, ridge, tol):
        """
        Exact M-step for one submodel's calibration vector beta_m: maximizes
        the concave, ridge-penalized weighted multinomial-logit
        log-likelihood (Eq. 8)

            Q_m(beta_m) = sum_i c_i [beta_{m,y_i} u_{i,y_i} - log sum_j exp(beta_{m,j} u_{i,j})]
                          - ridge * ||beta_m||^2

        to global optimality via Newton's method on the closed-form
        gradient/Hessian (Eqs. 10-11 of the derivation). The ridge term
        keeps beta_m finite under near-separable data. -H is positive
        definite for ridge > 0, so `-step` below
        is always an ascent direction; a short backtracking line search
        handles the fact that Q isn't exactly quadratic.

        u_m: [N, K] this submodel's log-probabilities (u_i^(m), fixed).
        c_m: [N] = gamma_{i,m} * nu_{y_i} (this submodel's E-step responsibility-weighted sample weights).
        """
        N, K = u_m.shape
        rows = np.arange(N)
        onehot_y = np.zeros((N, K))
        onehot_y[rows, y_idx] = 1.0

        def objective(b):
            z = b[None, :] * u_m
            z_max = z.max(axis=1, keepdims=True)
            logsumexp = z_max[:, 0] + np.log(np.exp(z - z_max).sum(axis=1))
            return float(np.sum(c_m * (z[rows, y_idx] - logsumexp)) - ridge * np.sum(b ** 2))

        beta = beta0.copy()
        Q = objective(beta)

        for _ in range(newton_iters):
            z = beta[None, :] * u_m
            pi = self._softmax(z)                                              # [N, K]

            grad = (c_m[:, None] * u_m * (onehot_y - pi)).sum(axis=0) - 2 * ridge * beta   # [K], Eq. 10

            # Hessian (Eq. 11), vectorized: H = V^T diag(c) V - diag(D) - 2*ridge*I,
            # with V_{i,k} = u_{i,k} pi_{i,k} and D_k = sum_i c_i u_{i,k}^2 pi_{i,k}.
            V = u_m * pi                                                        # [N, K]
            D = (c_m[:, None] * (u_m ** 2) * pi).sum(axis=0)                    # [K]
            H = (c_m[:, None] * V).T @ V - np.diag(D) - 2 * ridge * np.eye(K)

            step = np.linalg.solve(H, grad)

            # Backtracking: shrink the Newton step until it actually improves Q
            # (guards against the non-quadratic tail of the objective far from
            # the optimum; near convergence this loop exits after one try).
            new_beta, new_Q = beta - step, -np.inf
            for _ in range(10):
                new_Q = objective(new_beta)
                if new_Q >= Q:
                    break
                step /= 2
                new_beta = beta - step

            converged = abs(new_Q - Q) < tol * (abs(Q) + 1e-12)
            beta, Q = new_beta, new_Q
            if converged:
                break

        return beta

    def _fit_single(self, u, y_idx, nu_i, seed, max_iter, newton_iters, tol,
                     ridge, verbose, label=""):
        """
        One EM trajectory from a fresh random init. `u` ([N, M, K], fixed
        per-submodel log-probabilities) and `nu_i` ([N], nu_{y_i} per
        sample) are fixed throughout. Returns (alpha, beta, final_ll) --
        operates on local arrays, not self.alpha/self.beta, so multiple
        calls from fit() don't interfere with each other.
        """
        rng = np.random.default_rng(seed)
        N, M, K = u.shape

        # Randomized init, not an exact symmetric start (a perfectly
        # uniform/neutral init is itself a stationary point of this
        # non-convex outer problem) -- std=1 is deliberately wide so
        # different starts land in different basins, for fit()'s
        # keep-the-best-of-n_starts to matter.
        alpha = np.full(M, 1.0 / M)
        beta = rng.standard_normal((M, K))

        idx_n, idx_m = np.arange(N)[:, None], np.arange(M)[None, :]
        prev_ll = -np.inf
        ll = -np.inf

        for it in range(max_iter):
            # ---- E-step: responsibilities (Eq. estep) ----
            z = beta[None, :, :] * u
            tilde_p = self._softmax(z)                                          # [N, M, K]
            tilde_p_y = tilde_p[idx_n, idx_m, y_idx[:, None]]                    # [N, M] = tilde_p_{i,y_i}^{(m)}
            joint = alpha[None, :] * tilde_p_y                                   # [N, M]
            p_hat_y = joint.sum(axis=1, keepdims=True)                           # [N, 1] = p_hat_{i,y_i}
            # p_hat_y can legitimately underflow to exactly 0.0 in float64 (e.g.
            # early in EM, when a random beta start pushes beta_{m,y_i} u_{i,y_i}
            # very negative for every m at once) -- joint is then 0/0 for that
            # row rather than x/0, since every non-negative joint component
            # sums to p_hat_y. Clip the denominator so that resolves to gamma=0
            # (no submodel gets responsibility credit for a sample the model
            # currently assigns ~zero probability) instead of NaN, which would
            # otherwise poison weighted_gamma/alpha/beta for every later iter.
            gamma = joint / np.clip(p_hat_y, 1e-300, None)                       # [N, M]

            # Observed-data weighted log-likelihood (sum_i nu_yi log p_hat_iyi),
            # evaluated at the CURRENT (alpha, beta) before this iteration's
            # updates -- EM guarantees this is non-decreasing across iterations
            # WITHIN one trajectory (it says nothing about which stationary
            # point different starts land on -- that's what fit() compares).
            ll = float(np.sum(nu_i * np.log(np.clip(p_hat_y[:, 0], 1e-300, None))))

            # ---- M-step, alpha: closed form (Eq. alpha-update) ----
            weighted_gamma = nu_i[:, None] * gamma                               # [N, M] = c_i for each m
            alpha = weighted_gamma.sum(axis=0) / nu_i.sum()

            # ---- M-step, beta: exact per-submodel Newton/IRLS (Eq. beta-sub) ----
            for m in range(M):
                beta[m] = self._fit_beta_m(
                    u[:, m, :], y_idx, weighted_gamma[:, m], beta[m],
                    newton_iters=newton_iters, ridge=ridge, tol=tol,
                )

            if verbose and (it % 10 == 0 or it == max_iter - 1):
                print(f"{label}EM iter {it}: weighted log-lik = {ll:.4f}")

            if abs(ll - prev_ll) < tol * (abs(prev_ll) + 1e-12):
                if verbose:
                    print(f"{label}EM converged at iter {it} (delta log-lik = {ll - prev_ll:.2e})")
                break
            prev_ll = ll

        return alpha, beta, ll

    def fit(self, logits, y_idx, nu, n_starts=10, max_iter=200, newton_iters=25,
            tol=1e-6, ridge=1e-3, seed=42, verbose=True):
        """
        Fits alpha and {beta_m} by EM, from `n_starts` independent random
        initializations. `logits` ([N, M, K], raw submodel outputs) and
        `y_idx` ([N], integer true class per sample) are fixed throughout.

        Each M-step's beta_m-subproblem is solved to EXACT global
        optimality (concave, ridge-penalized multinomial-logit
        log-likelihood -- see EMEnsembleOptimizer's docstring). Only the
        outer (alpha, {beta_m}) coupling remains non-convex (still
        bilinear-type), so this still
        runs `n_starts` independent trajectories (different seeds) and
        keeps whichever converges to the highest final observed-data
        log-likelihood. Each trajectory is cheap on its own (Newton
        converges in a handful of iterations per M-step, no autodiff), so
        this multiplies fit()'s own cost by n_starts without repeating the
        (much more expensive) submodel logit extraction the caller already
        did once to produce `logits`.
        """
        N, M, K = logits.shape
        u = self._log_softmax(logits)  # u_i^(m): fixed for every start
        nu_i = nu[y_idx]  # [N], nu_{y_i}

        best_alpha, best_beta, best_ll = None, None, -np.inf
        for start in range(n_starts):
            label = f"[start {start + 1}/{n_starts}] " if verbose else ""
            alpha, beta, ll = self._fit_single(
                u, y_idx, nu_i, seed=seed + start,
                max_iter=max_iter, newton_iters=newton_iters, tol=tol,
                ridge=ridge, verbose=verbose, label=label,
            )
            if verbose:
                print(f"{label}final weighted log-lik = {ll:.4f}")
            if ll > best_ll:
                best_alpha, best_beta, best_ll = alpha, beta, ll

        self.alpha, self.beta = best_alpha, best_beta
        if verbose:
            print(f"Best of {n_starts} starts: weighted log-lik = {best_ll:.4f}")
        return self

class LinearStackingOptimizer:
    """
    Alternative ensemble combiner to EMEnsembleOptimizer: plain (ridge-
    regularized) least-squares stacking, a la Breiman/Wolpert stacked
    regression. No EM, no iterative optimization, no random restarts --
    meta-features are the M submodels' own predicted class probabilities,
    concatenated per sample into one [M*K] vector, and a single closed-form
    linear regression maps those meta-features onto one-hot(y). Selected via
    CoralFilterEnsembler's `ensemble_method` argument in filter.py.

    Model:
        p_i^(m)  = softmax(logit_i^(m))                        -- fixed submodel output
        phi_i    = concat_m(p_i^(m))  in R^{MK}                 -- stacked meta-features
        score_i  = W^T phi_i + b      in R^K                    -- W: [MK, K], b: [K]
        p_hat_i  = clip(score_i, 0) / sum(clip(score_i, 0))     -- projected back onto the simplex

    W, b are the *unique global optimum* of the ridge-penalized weighted
    least-squares problem

        min_{W,b} sum_i nu_{y_i} || (W^T phi_i + b) - onehot(y_i) ||^2 + ridge * ||W||_F^2

    solved once via the normal equations. The problem is convex/quadratic
    in (W, b), so unlike EMEnsembleOptimizer.fit there's no local-optimum
    risk and no benefit to multiple random starts.
    """

    def __init__(self, M, K, ridge=1e-3):
        self.M = M
        self.K = K
        self.ridge = ridge
        self.W = np.zeros((M * K, K))
        self.b = np.zeros(K)

    def _features(self, logits):
        """logits: [N, M, K] raw submodel logits -> phi: [N, M*K] stacked probs."""
        z = logits - logits.max(axis=-1, keepdims=True)
        e = np.exp(z)
        p = e / e.sum(axis=-1, keepdims=True)
        return p.reshape(len(logits), self.M * self.K)

    def fit(self, logits, y_idx, nu, seed=None, verbose=True):
        """
        Closed-form weighted ridge regression of one-hot(y) on stacked
        submodel probabilities. `seed` is accepted only for interface
        parity with EMEnsembleOptimizer.fit -- there's no randomness here
        to seed.
        """
        N = len(logits)
        phi = self._features(logits)                       # [N, MK]
        onehot_y = np.zeros((N, self.K))
        onehot_y[np.arange(N), y_idx] = 1.0
        w = nu[y_idx]                                       # [N], sample weights

        phi_aug = np.hstack([phi, np.ones((N, 1))])          # [N, MK+1], appended bias column
        sw = np.sqrt(w)[:, None]
        phi_w, y_w = phi_aug * sw, onehot_y * sw

        d = phi_aug.shape[1]
        reg = self.ridge * np.eye(d)
        reg[-1, -1] = 0  # never penalize the bias term

        A = phi_w.T @ phi_w + reg
        Bmat = phi_w.T @ y_w
        coef = np.linalg.solve(A, Bmat)                      # [MK+1, K]

        self.W, self.b = coef[:-1], coef[-1]

        if verbose:
            preds = self.predict_proba(logits)
            acc = float(np.mean(np.argmax(preds, axis=1) == y_idx))
            mse = float(np.mean(w[:, None] * (phi_aug @ coef - onehot_y) ** 2))
            print(f"Linear stacking fit: weighted training MSE = {mse:.4f}, training accuracy = {acc:.4f}")

        return self

    def predict_proba(self, logits):
        """logits: [N, M, K] raw submodel logits -> p_hat: [N, K]."""
        phi = self._features(logits)
        scores = np.clip(phi @ self.W + self.b[None, :], 0.0, None)
        totals = scores.sum(axis=1, keepdims=True)
        # A row can be all-zero if every clipped score underflows to 0 --
        # fall back to a uniform distribution over classes for that row
        # rather than dividing by zero.
        uniform = np.full_like(scores, 1.0 / self.K)
        return np.where(totals > 0, scores / np.clip(totals, 1e-300, None), uniform)

class AdamEnsembleOptimizer:
    """
    Alternative ensemble combiner to EMEnsembleOptimizer: maximizes the
    EXACT SAME objective -- the same weighted observed-data log-likelihood
    under the same multinomial-logit mixture model (see EMEnsembleOptimizer's
    docstring) -- via first-order gradient ascent (torch.optim.Adam) with
    autodiff, instead of EM + Newton/IRLS. Selected via
    CoralFilterEnsembler's `ensemble_method` argument in filter.py.

    Model (identical to EMEnsembleOptimizer's):
        p_i^(m)  = softmax(logit_i^(m))                                 -- fixed submodel output
        u_i^(m)  = log p_i^(m)                                          -- fixed per-submodel log-probabilities
        beta_m in R^K                                                    -- per-submodel, per-class calibration
        tilde_p_{i,k}^(m)(beta_m) = softmax_k(beta_{m,k} u_{i,k}^(m))
        p_hat_i  = sum_m alpha_m tilde_p_i^(m)(beta_m),   alpha in simplex

    Objective (same quantity EMEnsembleOptimizer's EM fit maximizes, just
    optimized differently -- no E-step/M-step split, no closed forms, no
    exact Newton solve; plain stochastic-gradient-style ascent on the whole
    (alpha, beta) vector at once):
        L(alpha, beta) = sum_i nu_{y_i} log p_hat_{i, y_i}

    `alpha` is optimized in an unconstrained R^M and mapped through softmax
    to stay on the simplex; `beta` is already unconstrained, same as in
    EMEnsembleOptimizer. Because Adam only has gradient (not curvature)
    information, this typically needs many more iterations than EM+Newton
    to reach a comparable optimum, and -- like EM -- the outer (alpha, beta)
    problem is non-convex, so multiple random restarts (`n_starts`) still
    matter here too.
    """

    def __init__(self, M, K):
        self.M = M
        self.K = K
        self.alpha = np.full(M, 1.0 / M)
        self.beta = np.ones((M, K))  # beta_m = 1 (all-ones) reproduces each submodel's own p_i^(m) unchanged

    @staticmethod
    def _softmax(logits):
        z = logits - logits.max(axis=-1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=-1, keepdims=True)

    @staticmethod
    def _log_softmax(logits):
        z = logits - logits.max(axis=-1, keepdims=True)
        return z - np.log(np.exp(z).sum(axis=-1, keepdims=True))

    def predict_proba(self, logits):
        """logits: [N, M, K] raw submodel logits -> p_hat: [N, K]. Identical
        formula to EMEnsembleOptimizer.predict_proba, evaluated in NumPy
        against the fitted self.alpha/self.beta."""
        u = self._log_softmax(logits)
        z = self.beta[None, :, :] * u
        tilde_p = self._softmax(z)
        return (self.alpha[None, :, None] * tilde_p).sum(axis=1)

    @staticmethod
    def _weighted_ll(alpha, beta, u, y_idx_t, nu_i, idx):
        """torch, no_grad: weighted mean log-lik of (alpha, beta) on (u, y_idx_t, nu_i)."""
        z = beta[None, :, :] * u
        tilde_p = torch.softmax(z, dim=-1)
        tilde_p_y = tilde_p.gather(2, idx).squeeze(2)
        p_hat_y = (alpha[None, :] * tilde_p_y).sum(dim=1)
        return float((nu_i * torch.log(p_hat_y.clamp_min(1e-300))).mean())

    def _fit_single(self, u, y_idx_t, nu_i, seed, max_iter, lr, tol, patience, verbose, label="",
                     u_val=None, y_val_idx_t=None, nu_i_val=None):
        """
        One Adam trajectory from a fresh random init. `u` ([N, M, K] torch
        tensor, fixed per-submodel log-probabilities) and `y_idx_t`/`nu_i`
        are fixed throughout. Returns (best_alpha, best_beta, best_ll) as
        NumPy arrays/float -- operates on local tensors, not
        self.alpha/self.beta, so multiple calls from fit() don't interfere
        with each other.

        Unlike EMEnsembleOptimizer's EM updates, Adam's log-likelihood is
        NOT guaranteed to increase monotonically step to step (no proof of
        ascent the way EM's E/M-step decomposition gives you), so a single
        below-tolerance step isn't reliable evidence of convergence -- it
        could just be a momentary wobble. For the same reason, the BEST
        iterate seen over the whole trajectory is tracked and returned --
        not just wherever the loop happens to stop.

        Two model-selection/stopping modes, chosen by whether a validation
        set is passed:

        - u_val/y_val_idx_t/nu_i_val given: validation-based early stopping,
          matching the original gradient-descent `EnsembleOptimizer`'s
          train_weights() -- "best" and "converged" are both judged by
          weighted log-lik on this HELD-OUT set (an iteration counts as
          `patience` iterations closer to stopping whenever it fails to
          beat the best validation log-lik seen so far). This is a
          different, and stronger, guard against overfitting THIS fit
          itself than the no-validation mode below, at the cost of
          consuming a held-out split for early stopping rather than only
          for a final post-hoc check (see CoralFilterEnsembler.train_ensemble).
        - Omitted (default): convergence requires `patience` consecutive
          iterations where the TRAINING log-lik changes by less than `tol`
          (relative) -- the same "run until convergence" idea, just judged
          against the training objective directly since there's no held-out
          set in this mode. `max_iter` is a safety cap either way, in case
          neither stopping condition is ever met.
        """
        torch.manual_seed(seed)
        N, M, K = u.shape
        use_val = u_val is not None

        # Same init convention as EMEnsembleOptimizer._fit_single: alpha
        # starts uniform (raw_alpha=0 -> softmax is uniform), beta starts
        # from a wide random normal so different starts land in different
        # basins of the non-convex outer problem.
        raw_alpha = torch.zeros(M, dtype=u.dtype, requires_grad=True)
        beta = torch.randn(M, K, dtype=u.dtype, requires_grad=True)

        optimizer = torch.optim.Adam([raw_alpha, beta], lr=lr)
        idx = y_idx_t.view(-1, 1, 1).expand(-1, M, 1)  # [N, M, 1], for gathering tilde_p at the true class
        idx_val = y_val_idx_t.view(-1, 1, 1).expand(-1, M, 1) if use_val else None

        best_ll = -float("inf")
        best_alpha_np, best_beta_np = None, None
        prev_ll = -float("inf")
        stable_iters = 0
        for it in range(max_iter):
            optimizer.zero_grad()
            alpha = torch.softmax(raw_alpha, dim=0)
            z = beta[None, :, :] * u
            tilde_p = torch.softmax(z, dim=-1)                              # [N, M, K]
            tilde_p_y = tilde_p.gather(2, idx).squeeze(2)                   # [N, M]
            p_hat_y = (alpha[None, :] * tilde_p_y).sum(dim=1)               # [N]
            loss = -(nu_i * torch.log(p_hat_y.clamp_min(1e-300))).sum()
            loss.backward()
            optimizer.step()

            train_ll = -float(loss.detach())
            with torch.no_grad():
                if use_val:
                    alpha_cur = torch.softmax(raw_alpha, dim=0)
                    ll = self._weighted_ll(alpha_cur, beta, u_val, y_val_idx_t, nu_i_val, idx_val)
                else:
                    ll = train_ll

            improved = ll > best_ll
            if improved:
                best_ll = ll
                with torch.no_grad():
                    best_alpha_np = torch.softmax(raw_alpha, dim=0).numpy()
                    best_beta_np = beta.numpy().copy()

            if verbose and (it % 10 == 0 or it == max_iter - 1):
                tag = "val" if use_val else "train"
                print(f"{label}Adam iter {it}: {tag} weighted log-lik = {ll:.4f} (best = {best_ll:.4f})")

            if use_val:
                stable_iters = 0 if improved else stable_iters + 1
                if stable_iters > patience:
                    if verbose:
                        print(f"{label}Adam early-stopped at iter {it} "
                              f"({patience} iters without a validation log-lik improvement)")
                    break
            else:
                if abs(train_ll - prev_ll) < tol * (abs(prev_ll) + 1e-12):
                    stable_iters += 1
                    if stable_iters >= patience:
                        if verbose:
                            print(f"{label}Adam converged at iter {it} "
                                  f"(delta log-lik < {tol:.0e} for {patience} consecutive iters)")
                        break
                else:
                    stable_iters = 0
                prev_ll = train_ll

        return best_alpha_np, best_beta_np, best_ll

    def fit(self, logits, y_idx, nu, n_starts=10, max_iter=20000, lr=0.05, tol=1e-6, patience=20, seed=42, verbose=True,
            val_logits=None, val_y_idx=None):
        """
        Fits alpha and {beta_m} by Adam gradient ascent on the SAME
        objective EMEnsembleOptimizer.fit maximizes (see class docstring),
        from `n_starts` independent random initializations -- the best of
        these by final weighted log-likelihood is kept, same selection rule
        as EMEnsembleOptimizer.fit.

        Each trajectory runs until convergence rather than for a fixed
        iteration count -- see _fit_single for the two stopping modes.
        `max_iter` is just a safety cap that should rarely if ever be hit.

        logits: [N, M, K] raw submodel outputs (NumPy). y_idx: [N] integer
        true class per sample. nu: [K] per-class sample weights.

        val_logits/val_y_idx (optional, NumPy, same shapes as logits/y_idx
        but a disjoint held-out set): if given, switches every trajectory to
        validation-based early stopping/model-selection (matching the
        original gradient-descent `EnsembleOptimizer`'s train_weights())
        instead of judging convergence and "best" purely from the training
        objective. See _fit_single's docstring for the tradeoff.
        """
        u_np = self._log_softmax(logits)
        u = torch.from_numpy(u_np).double()
        y_idx_t = torch.from_numpy(np.asarray(y_idx)).long()
        nu_i = torch.from_numpy(np.asarray(nu)[y_idx]).double()

        if val_logits is not None:
            u_val = torch.from_numpy(self._log_softmax(val_logits)).double()
            y_val_idx_t = torch.from_numpy(np.asarray(val_y_idx)).long()
            nu_i_val = torch.from_numpy(np.asarray(nu)[val_y_idx]).double()
        else:
            u_val, y_val_idx_t, nu_i_val = None, None, None

        best_alpha, best_beta, best_ll = None, None, -np.inf
        for start in range(n_starts):
            label = f"[start {start + 1}/{n_starts}] " if verbose else ""
            alpha, beta, ll = self._fit_single(
                u, y_idx_t, nu_i, seed=seed + start,
                max_iter=max_iter, lr=lr, tol=tol, patience=patience, verbose=verbose, label=label,
                u_val=u_val, y_val_idx_t=y_val_idx_t, nu_i_val=nu_i_val,
            )
            if verbose:
                print(f"{label}final weighted log-lik = {ll:.4f}")
            if ll > best_ll:
                best_alpha, best_beta, best_ll = alpha, beta, ll

        self.alpha, self.beta = best_alpha, best_beta
        if verbose:
            print(f"Best of {n_starts} starts: weighted log-lik = {best_ll:.4f}")
        return self

class NeuralNetEnsembleOptimizer:
    """
    Alternative ensemble combiner: a small feedforward network stacked on
    top of the M submodels' own predicted class probabilities -- the SAME
    meta-features LinearStackingOptimizer uses (phi_i = concat_m(p_i^(m))
    in R^{MK}) -- but with a nonlinear multi-layer head instead of a single
    closed-form linear map, fit by gradient descent (torch.optim.Adam) on
    the nu-weighted categorical cross-entropy loss:

        phi_i    = concat_m(softmax(logit_i^(m)))  in R^{MK}
        p_hat_i  = softmax(f_theta(phi_i))          in R^K
        L(theta) = -sum_i nu_{y_i} log p_hat_{i,y_i}(theta) / sum_i nu_{y_i}

    -- the same nu-weighted objective every other ensemble_method in this
    module maximizes the log-lik form of (see EMEnsembleOptimizer's
    docstring), just with a plain multi-layer classifier instead of a
    mixture-of-reweighted-experts structure: no alpha, no per-submodel
    tilde_p, just phi_i -> hidden layers -> K-way softmax. Selected via
    CoralFilterEnsembler's `ensemble_method` argument in filter.py.

    Layer weights/biases are plain NumPy arrays, named W0/b0, W1/b1, ...
    (not packed into one blob) so CoralFilterEnsembler's existing save/load
    machinery -- a flat list of named-array attributes per ensemble_method,
    see filter.py's _ENSEMBLE_PARAM_NAMES -- works for this method exactly
    like every other one, with no special-casing.

    Same non-convexity/initialization-sensitivity as AdamEnsembleOptimizer,
    so fit() supports the same multiple random restarts (n_starts) and
    held-out early stopping.
    """

    def __init__(self, M, K, hidden_sizes=(32, 16)):
        self.M = M
        self.K = K
        self.hidden_sizes = tuple(hidden_sizes)
        sizes = (M * K,) + self.hidden_sizes + (K,)
        self.n_layers = len(sizes) - 1
        # Placeholder init (overwritten by fit()) -- just needs to exist with
        # the right shapes/names for _ENSEMBLE_PARAM_NAMES-based save/load
        # to work before the first fit() call.
        rng = np.random.default_rng(0)
        for i in range(self.n_layers):
            fan_in, fan_out = sizes[i], sizes[i + 1]
            setattr(self, f"W{i}", (rng.standard_normal((fan_in, fan_out)) * np.sqrt(2.0 / fan_in)))
            setattr(self, f"b{i}", np.zeros(fan_out))

    @staticmethod
    def _features(logits):
        """logits: [N, M, K] raw submodel logits -> phi: [N, M*K] stacked
        probs -- identical convention to LinearStackingOptimizer._features."""
        z = logits - logits.max(axis=-1, keepdims=True)
        e = np.exp(z)
        p = e / e.sum(axis=-1, keepdims=True)
        return p.reshape(len(logits), -1)

    def _forward(self, phi, weights):
        """phi: [N, MK] torch tensor. weights: flat [W0,b0,W1,b1,...] torch
        tensors. Returns pre-softmax logits [N, K]. ReLU between hidden
        layers, none after the final (output) layer."""
        x = phi
        for i in range(self.n_layers):
            W, b = weights[2 * i], weights[2 * i + 1]
            x = x @ W + b
            if i < self.n_layers - 1:
                x = torch.relu(x)
        return x

    def predict_proba(self, logits):
        """logits: [N, M, K] raw submodel logits -> p_hat: [N, K]."""
        phi = torch.from_numpy(self._features(logits)).double()
        weights = []
        for i in range(self.n_layers):
            weights.append(torch.from_numpy(np.asarray(getattr(self, f"W{i}"))).double())
            weights.append(torch.from_numpy(np.asarray(getattr(self, f"b{i}"))).double())
        with torch.no_grad():
            out = self._forward(phi, weights)
            proba = torch.softmax(out, dim=-1)
        return proba.numpy()

    def _init_weights(self, seed):
        rng = np.random.default_rng(seed)
        sizes = (self.M * self.K,) + self.hidden_sizes + (self.K,)
        weights = []
        for i in range(self.n_layers):
            fan_in, fan_out = sizes[i], sizes[i + 1]
            # He init (ReLU hidden layers); output layer gets the same
            # scale for simplicity, harmless since it's linear->softmax.
            W = torch.tensor(rng.standard_normal((fan_in, fan_out)) * np.sqrt(2.0 / fan_in),
                              dtype=torch.float64, requires_grad=True)
            b = torch.zeros(fan_out, dtype=torch.float64, requires_grad=True)
            weights.append(W)
            weights.append(b)
        return weights

    def _weighted_mean_ll(self, phi, weights, y_idx_t, nu_i, nu_sum):
        out = self._forward(phi, weights)
        logp = torch.log_softmax(out, dim=-1)
        logp_y = logp.gather(1, y_idx_t.view(-1, 1)).squeeze(1)
        return (nu_i * logp_y).sum() / nu_sum

    def _fit_single(self, phi, y_idx_t, nu_i, nu_sum, seed, max_iter, lr, weight_decay, tol, patience,
                     verbose, label="", phi_val=None, y_val_idx_t=None, nu_i_val=None, nu_sum_val=None):
        """
        One Adam trajectory from a fresh random init. Mirrors
        AdamEnsembleOptimizer._fit_single's structure exactly: the BEST
        iterate seen over the whole trajectory is tracked and returned (Adam
        gives no monotonic-ascent guarantee), and stopping is either
        validation-based (if phi_val given) or training-objective-based.
        """
        torch.manual_seed(seed)
        weights = self._init_weights(seed)
        optimizer = torch.optim.Adam(weights, lr=lr, weight_decay=weight_decay)
        use_val = phi_val is not None

        best_ll = -float("inf")
        best_weights_np = None
        prev_ll = -float("inf")
        stable_iters = 0

        for it in range(max_iter):
            optimizer.zero_grad()
            loss = -self._weighted_mean_ll(phi, weights, y_idx_t, nu_i, nu_sum)
            loss.backward()
            optimizer.step()

            train_ll = -float(loss.detach())
            with torch.no_grad():
                if use_val:
                    ll = float(self._weighted_mean_ll(phi_val, weights, y_val_idx_t, nu_i_val, nu_sum_val))
                else:
                    ll = train_ll

            improved = ll > best_ll
            if improved:
                best_ll = ll
                best_weights_np = [w.detach().numpy().copy() for w in weights]

            if verbose and (it % 50 == 0 or it == max_iter - 1):
                tag = "val" if use_val else "train"
                print(f"{label}NN iter {it}: {tag} weighted mean log-lik = {ll:.4f} (best = {best_ll:.4f})")

            if use_val:
                stable_iters = 0 if improved else stable_iters + 1
                if stable_iters > patience:
                    if verbose:
                        print(f"{label}NN early-stopped at iter {it} "
                              f"({patience} iters without a validation log-lik improvement)")
                    break
            else:
                if abs(train_ll - prev_ll) < tol * (abs(prev_ll) + 1e-12):
                    stable_iters += 1
                    if stable_iters >= patience:
                        if verbose:
                            print(f"{label}NN converged at iter {it} "
                                  f"(delta log-lik < {tol:.0e} for {patience} consecutive iters)")
                        break
                else:
                    stable_iters = 0
                prev_ll = train_ll

        return best_weights_np, best_ll

    def fit(self, logits, y_idx, nu, n_starts=10, max_iter=5000, lr=1e-3, weight_decay=1e-4,
            tol=1e-6, patience=20, seed=42, verbose=True, val_logits=None, val_y_idx=None):
        """
        logits: [N, M, K] raw submodel outputs (NumPy). y_idx: [N] integer
        true class per sample. nu: [K] per-class sample weights.

        val_logits/val_y_idx (optional, NumPy, same shapes but a disjoint
        held-out set): if given, switches every trajectory to
        validation-based early stopping/model-selection, same as
        AdamEnsembleOptimizer.fit -- see _fit_single's docstring.
        """
        phi = torch.from_numpy(self._features(logits)).double()
        y_idx_t = torch.from_numpy(np.asarray(y_idx)).long()
        nu_arr = np.asarray(nu)
        nu_i = torch.from_numpy(nu_arr[y_idx]).double()
        nu_sum = float(nu_i.sum())

        if val_logits is not None:
            phi_val = torch.from_numpy(self._features(val_logits)).double()
            y_val_idx_t = torch.from_numpy(np.asarray(val_y_idx)).long()
            nu_i_val = torch.from_numpy(nu_arr[val_y_idx]).double()
            nu_sum_val = float(nu_i_val.sum())
        else:
            phi_val, y_val_idx_t, nu_i_val, nu_sum_val = None, None, None, None

        best_weights, best_ll = None, -np.inf
        for start in range(n_starts):
            label = f"[start {start + 1}/{n_starts}] " if verbose else ""
            weights_np, ll = self._fit_single(
                phi, y_idx_t, nu_i, nu_sum, seed=seed + start,
                max_iter=max_iter, lr=lr, weight_decay=weight_decay, tol=tol, patience=patience,
                verbose=verbose, label=label,
                phi_val=phi_val, y_val_idx_t=y_val_idx_t, nu_i_val=nu_i_val, nu_sum_val=nu_sum_val,
            )
            if verbose:
                print(f"{label}final weighted mean log-lik = {ll:.4f}")
            if ll > best_ll:
                best_weights, best_ll = weights_np, ll

        for i in range(self.n_layers):
            setattr(self, f"W{i}", best_weights[2 * i])
            setattr(self, f"b{i}", best_weights[2 * i + 1])

        if verbose:
            print(f"Best of {n_starts} starts: weighted mean log-lik = {best_ll:.4f}")
        return self


class ClassReweightingOptimizer:
    """
    Each submodel's own predicted probabilities are reweighted per-class by
    a strictly positive vector w_m, renormalized back onto the simplex, and
    the M reweighted-and-renormalized submodel distributions are combined by
    a convex combination alpha in Delta^{M-1}.

    Model (Eqs. 2-3):
        p_i^(m)       = softmax(logit_i^(m))                                -- fixed submodel output
        tilde_p_i^(m) = (w_m ⊙ p_i^(m)) / (1^T (w_m ⊙ p_i^(m)))             -- Eq. 3, Hadamard-reweight + renormalize
        p_hat_i       = sum_m alpha_m tilde_p_i^(m)                        -- Eq. 2
        subject to    alpha in Delta^{M-1},   w_m in R^K_{++},  m=1..M

    Eq. 2 is exactly the marginal of a latent-class mixture (P(z_i=m)=alpha_m,
    P(y_i=k | z_i=m)=tilde_p_{i,k}^{(m)}(w_m)), so this is fit by EM.
    Weighted log-likelihood MAXIMIZED (same quantity EMEnsembleOptimizer/
    AdamEnsembleOptimizer maximize, and nu_k plays the same class-weighting
    role as `nu` elsewhere in this module):
        L(alpha, W) = sum_i nu_{y_i} log p_hat_{i, y_i}

    E-step (responsibilities):
        gamma_{im} = alpha_m tilde_p_{i,y_i}^{(m)}(w_m) / sum_{m'} alpha_{m'} tilde_p_{i,y_i}^{(m')}(w_{m'})

    M-step, alpha (closed form, exact):
        alpha_m <- sum_i nu_{y_i} gamma_{im} / sum_i nu_{y_i}

    M-step, W (per submodel m, independent): maximizing
        Q_m(w) = sum_k n_k log w_k - sum_i c_i log(sum_j w_j p_{i,j})),  c_i := gamma_{im} nu_{y_i},  n_k := sum_{i: y_i=k} c_i
    is non-concave (a concave term minus a convex term), so instead of an
    exact solve it's improved via a few Minorize-Maximize (MM) fixed-point
    iterations per M-step (a valid Generalized-EM update -- any step that
    increases Q_m, not just an exact maximizer, preserves EM's monotonic
    ascent guarantee on the observed-data log-likelihood):
        s_i    <- sum_j w_j p_{i,j}                          (current fit)
        d_k    <- sum_i c_i p_{i,k} / s_i
        w_k    <- n_k / d_k                                  (closed form, from minorizing -log(s_i(w)) by its tangent line at the current w)
    This is the same multiplicative fixed-point family as iterative
    proportional fitting / generalized iterative scaling -- no learning
    rate, gradient, or autodiff. A small pseudocount is added to n_k so a
    zero-count class doesn't force w_k -> 0, and w_m is renormalized
    (divided by its sum) after each M-step since Eq. 3 is invariant to
    w_m -> c*w_m for any c > 0 (otherwise w_m would drift unboundedly
    across iterations without changing any prediction).

    Only the OUTER (alpha, {w_m}) coupling is non-convex -- each M-step
    improves Q -- so fit() runs `n_starts` independent random restarts and
    keeps whichever converges to the highest final observed-data
    log-likelihood. This ascent is monotonic within a trajectory, so no
    held-out early stopping is needed (unlike the Adam-fit optimizers
    elsewhere in this module) -- see CoralFilterEnsembler.train_ensemble in
    filter.py. Selected via CoralFilterEnsembler's `ensemble_method`
    argument in filter.py.
    """

    def __init__(self, M, K):
        self.M = M
        self.K = K
        self.alpha = np.full(M, 1.0 / M)
        self.W = np.ones((M, K))  # w_m = 1 (all-ones) reproduces each submodel's own p_i^(m) unchanged

    @staticmethod
    def _softmax(logits):
        z = logits - logits.max(axis=-1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=-1, keepdims=True)

    def predict_proba(self, logits):
        """logits: [N, M, K] raw submodel logits -> p_hat: [N, K] (Eqs. 2-3)."""
        p = self._softmax(logits)                                    # [N, M, K]
        weighted = self.W[None, :, :] * p
        tilde_p = weighted / np.clip(weighted.sum(axis=-1, keepdims=True), 1e-300, None)
        return (self.alpha[None, :, None] * tilde_p).sum(axis=1)

    @staticmethod
    def _tilde_p_y(p, y_idx, W):
        """p: [N, M, K] fixed submodel probabilities, W: [M, K] current
        reweight vectors -> tilde_p_{i,y_i}^{(m)} (Eq. 3), shape [N, M]."""
        N, M, K = p.shape
        weighted = W[None, :, :] * p                                        # [N, M, K]
        tilde_p = weighted / np.clip(weighted.sum(axis=-1, keepdims=True), 1e-300, None)
        rows, ms = np.arange(N)[:, None], np.arange(M)[None, :]
        return tilde_p[rows, ms, y_idx[:, None]]                            # [N, M]

    def _e_step(self, p, y_idx, alpha, W):
        """Eq. estep: responsibilities gamma [N, M] and p_hat_{i,y_i} [N],
        both evaluated at the CURRENT (alpha, W)."""
        tilde_p_y = self._tilde_p_y(p, y_idx, W)                            # [N, M]
        joint = alpha[None, :] * tilde_p_y                                  # [N, M]
        p_hat_y = joint.sum(axis=1, keepdims=True)                          # [N, 1]
        gamma = joint / np.clip(p_hat_y, 1e-300, None)
        return gamma, p_hat_y[:, 0]

    def _mm_update_w(self, p_m, c_i, n_k, w0, mm_iters, pseudocount):
        """A few MM fixed-point iterations of Eq. mm-update for one
        submodel's w_m, starting from its current value w0 (a GEM step --
        need not run to convergence for EM's ascent guarantee to hold).
        p_m: [N, K] this submodel's own softmax probabilities (fixed).
        c_i: [N] = gamma_{im} * nu_{y_i} (this M-step's fixed weights).
        n_k: [K] = sum_{i: y_i=k} c_i (fixed for this M-step, precomputed
        by the caller since it doesn't depend on w)."""
        w = w0.copy()
        n_k = n_k + pseudocount
        for _ in range(mm_iters):
            s = np.clip((w[None, :] * p_m).sum(axis=1), 1e-300, None)       # [N]
            d = (p_m * (c_i / s)[:, None]).sum(axis=0)                      # [K]
            w = n_k / np.clip(d, 1e-300, None)
        return w / np.clip(w.sum(), 1e-300, None)                           # renormalize (scale non-identifiability)

    def _fit_single(self, p, y_idx, nu_i, seed, max_iter, mm_iters, tol, pseudocount, verbose, label="", init_alpha=None, init_W=None):
        """
        One EM trajectory, from a fresh random init by default, or from
        (init_alpha, init_W) when given -- e.g. a previous/reference fit's
        converged parameters, used by fit()'s `init` argument to seed one
        trajectory deterministically alongside its random restarts. `p`
        ([N, M, K], fixed per-submodel softmax probabilities) and
        `y_idx`/`nu_i` are fixed throughout. Returns (alpha, W, final_ll) --
        operates on local arrays, not self.alpha/self.W, so multiple calls
        from fit() don't interfere with each other. Mirrors
        EMEnsembleOptimizer._fit_single's structure exactly (random init,
        E-step/M-step loop, training-log-lik-plateau convergence).
        """
        rng = np.random.default_rng(seed)
        N, M, K = p.shape

        if init_alpha is None:
            alpha = np.full(M, 1.0 / M)
        else:
            alpha = init_alpha.copy()
        if init_W is None:
            # Randomized positive init (not the all-ones "no reweighting"
            # point, which would be the same start for every restart -- see
            # EMEnsembleOptimizer._fit_single's analogous beta init).
            W = np.exp(rng.standard_normal((M, K)))
        else:
            W = init_W.copy()

        rows = np.arange(N)
        onehot_y = np.zeros((N, K))
        onehot_y[rows, y_idx] = 1.0

        prev_ll = -np.inf
        ll = -np.inf

        for it in range(max_iter):
            # ---- E-step ----
            gamma, p_hat_y = self._e_step(p, y_idx, alpha, W)

            # Observed-data weighted log-likelihood at the CURRENT (alpha, W),
            # before this iteration's updates -- non-decreasing across
            # iterations within one trajectory (same guarantee as
            # EMEnsembleOptimizer's EM fit; different starts can still land
            # on different stationary points, which is what fit() compares).
            ll = float(np.sum(nu_i * np.log(np.clip(p_hat_y, 1e-300, None))))

            # ---- M-step, alpha: closed form ----
            weighted_gamma = nu_i[:, None] * gamma                          # [N, M] = c_i for each m
            alpha = weighted_gamma.sum(axis=0) / nu_i.sum()

            # ---- M-step, W: a few MM fixed-point iterations per submodel ----
            for m in range(M):
                c_i = weighted_gamma[:, m]                                  # [N]
                n_k = onehot_y.T @ c_i                                      # [K]
                W[m] = self._mm_update_w(p[:, m, :], c_i, n_k, W[m], mm_iters, pseudocount)

            if verbose and (it % 10 == 0 or it == max_iter - 1):
                print(f"{label}EM iter {it}: weighted log-lik = {ll:.4f}")

            if abs(ll - prev_ll) < tol * (abs(prev_ll) + 1e-12):
                if verbose:
                    print(f"{label}EM converged at iter {it} (delta log-lik = {ll - prev_ll:.2e})")
                break
            prev_ll = ll

        return alpha, W, ll

    def fit(self, logits, y_idx, nu, n_starts=10, max_iter=200, mm_iters=5, tol=1e-6, pseudocount=1e-6, seed=42, verbose=True, init=None):
        """
        Fits alpha and W by EM (closed-form alpha M-step, MM-fixed-point W
        M-step -- see class docstring), from `n_starts` independent random
        initializations -- the best of these by final weighted observed-
        data log-likelihood is kept, same selection rule as
        EMEnsembleOptimizer.fit.

        logits: [N, M, K] raw submodel outputs (NumPy). y_idx: [N] integer
        true class per sample. nu: [K] per-class sample weights.
        mm_iters: number of MM fixed-point iterations run per submodel,
        per EM iteration, for the (non-concave) W M-step -- a GEM step, not
        an exact solve, so this need not be large for the ascent guarantee
        to hold. pseudocount: added to each class's n_k before the w_k =
        n_k / d_k update, so a class with zero responsibility-weighted
        samples doesn't force w_k -> 0.
        init (tuple(alpha0, W0), optional): an EXTRA starting point run to
            convergence ALONGSIDE the `n_starts` random restarts, not
            instead of them -- every trajectory's final log-lik is compared
            and the best kept. EM/MM's per-trajectory ascent guarantee (see
            class docstring) means this trajectory's own final log-lik is
            guaranteed >= the log-lik `init` itself achieves on this fit's
            (logits, y_idx, nu).
        """
        N, M, K = logits.shape
        p = self._softmax(logits)  # p_i^(m): fixed for every start
        nu_i = nu[y_idx]  # [N], nu_{y_i}

        best_alpha, best_W, best_ll = None, None, -np.inf
        for start in range(n_starts):
            label = f"[start {start + 1}/{n_starts}] " if verbose else ""
            alpha, W, ll = self._fit_single(
                p, y_idx, nu_i, seed=seed + start,
                max_iter=max_iter, mm_iters=mm_iters, tol=tol, pseudocount=pseudocount,
                verbose=verbose, label=label,
            )
            if verbose:
                print(f"{label}final weighted log-lik = {ll:.4f}")
            if ll > best_ll:
                best_alpha, best_W, best_ll = alpha, W, ll

        if init is not None:
            init_alpha, init_W = init
            label = "[init] " if verbose else ""
            alpha, W, ll = self._fit_single(
                p, y_idx, nu_i, seed=seed,
                max_iter=max_iter, mm_iters=mm_iters, tol=tol, pseudocount=pseudocount,
                verbose=verbose, label=label, init_alpha=init_alpha, init_W=init_W,
            )
            if verbose:
                print(f"{label}final weighted log-lik = {ll:.4f}")
            if ll > best_ll:
                best_alpha, best_W, best_ll = alpha, W, ll

        self.alpha, self.W = best_alpha, best_W
        if verbose:
            n_trials = n_starts + (1 if init is not None else 0)
            print(f"Best of {n_trials} start(s) ({n_starts} random" +
                  (" + 1 given init)" if init is not None else ")") + f": weighted log-lik = {best_ll:.4f}")
        return self

class MultinomialRegressionOptimizer:
    """
    Fits ONLY the per-submodel, per-class calibration vectors beta_m -- no
    mixture weights alpha at all. Whereas EMEnsembleOptimizer combines M
    per-submodel calibrated DISTRIBUTIONS via a convex combination
    (p_hat_i = sum_m alpha_m tilde_p_i^(m)(beta_m)), this model instead
    pools all M submodels' calibrated LOGITS into a single joint
    multinomial logistic regression with one shared softmax at the end:

        p_i^(m)  = softmax(logit_i^(m))                       -- fixed submodel output
        u_i^(m)  = log p_i^(m)                                 -- fixed per-submodel log-probabilities
        beta_m in R^K                                          -- per-submodel, per-class calibration
        z_{i,k}  = sum_m beta_{m,k} u_{i,k}^(m)                -- summed calibrated logit for class k
        p_hat_i  = softmax(z_i)

    Dropping alpha (and the per-submodel softmax it was mixing) removes the
    bilinear (alpha, beta) coupling that makes EMEnsembleOptimizer's OUTER
    problem non-convex. What's left is a single, ordinary multinomial
    logistic regression in beta (with per-class-restricted "features"
    u_{i,k}^(m) -- class k's score only ever sees class k's log-probs from
    each submodel), which is EXACTLY CONCAVE once ridge-penalized
    (log-sum-exp of an affine map is convex, so its negative is concave).
    That means there is a UNIQUE global optimum, found by ONE exact
    Newton/IRLS solve -- no EM outer loop, no multiple random restarts, no
    non-convexity to search around (unlike EMEnsembleOptimizer,
    AdamEnsembleOptimizer, and ClassReweightingOptimizer, all of which keep
    an alpha-beta or alpha-W coupling and so need multi-start).

    Objective (maximized; same nu-weighting convention as elsewhere here):
        Q(beta) = sum_i nu_{y_i} [z_{i,y_i} - log sum_k exp(z_{i,k})] - ridge * ||beta||^2

    Selected via CoralFilterEnsembler's `ensemble_method` argument in
    filter.py.
    """

    def __init__(self, M, K):
        self.M = M
        self.K = K
        self.beta = np.ones((M, K))  # beta_m = 1 (all-ones) reproduces softmax(sum_m u^(m)), the plain sum-of-log-probs baseline

    @staticmethod
    def _softmax(logits):
        z = logits - logits.max(axis=-1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=-1, keepdims=True)

    @staticmethod
    def _log_softmax(logits):
        z = logits - logits.max(axis=-1, keepdims=True)
        return z - np.log(np.exp(z).sum(axis=-1, keepdims=True))

    def predict_proba(self, logits):
        """logits: [N, M, K] raw submodel logits -> p_hat: [N, K]."""
        u = self._log_softmax(logits)
        z = (self.beta[None, :, :] * u).sum(axis=1)   # [N, K]
        return self._softmax(z)

    def fit(self, logits, y_idx, nu, newton_iters=50, ridge=1e-3, tol=1e-10, seed=None, verbose=True):
        """
        Fits beta to global optimality via Newton's method on the exact
        gradient/Hessian of the objective above (see class docstring for
        why this is possible in one exact solve, unlike the other
        ensemble_method options). `seed` is accepted only for interface
        parity with the other optimizers' fit() -- there's no randomness
        here (beta is initialized at the fixed all-ones "no calibration"
        point; with a unique global optimum, init only affects the
        convergence path, never the answer).

        logits: [N, M, K] raw submodel outputs (NumPy). y_idx: [N] integer
        true class per sample. nu: [K] per-class sample weights.
        """
        N, M, K = logits.shape
        u = self._log_softmax(logits)
        c = np.asarray(nu)[y_idx]                          # [N]
        rows = np.arange(N)
        onehot_y = np.zeros((N, K))
        onehot_y[rows, y_idx] = 1.0

        def compute_z(theta):
            return (theta[None, :, :] * u).sum(axis=1)      # [N, K]

        def objective(theta):
            z = compute_z(theta)
            z_max = z.max(axis=1, keepdims=True)
            logsumexp = z_max[:, 0] + np.log(np.exp(z - z_max).sum(axis=1))
            return float(np.sum(c * (z[rows, y_idx] - logsumexp)) - ridge * np.sum(theta ** 2))

        theta = np.ones((M, K))
        Q = objective(theta)
        MK = M * K

        for it in range(newton_iters):
            z = compute_z(theta)
            pi = self._softmax(z)                            # [N, K]

            A = c[:, None] * (onehot_y - pi)                  # [N, K]
            grad = np.einsum('imk,ik->mk', u, A) - 2 * ridge * theta   # [M, K]

            # Full (M,K,M,K) Hessian, vectorized -- same "diag(pi) - pi pi^T"
            # concavity structure as EMEnsembleOptimizer._fit_beta_m's
            # per-submodel Hessian, but now with class k's "feature" being
            # u_{i,:,k} (the K-vector of that class's log-probs across all M
            # submodels) rather than a single submodel's u_{i,k}, since
            # dropping alpha means all M submodels' calibrations interact
            # through one shared softmax instead of M separate ones.
            sqrt_c_pi = np.sqrt(np.clip(c[:, None] * pi, 0, None))       # [N, K]
            C = sqrt_c_pi[:, None, :] * u                                # [N, M, K]
            block_diag = np.zeros((M, K, M, K))
            for k in range(K):
                Ck = C[:, :, k]                                          # [N, M]
                block_diag[:, k, :, k] -= Ck.T @ Ck

            B = np.sqrt(c)[:, None, None] * pi[:, None, :] * u           # [N, M, K]
            B_flat = B.reshape(N, MK)
            cross_term = (B_flat.T @ B_flat).reshape(M, K, M, K)

            H = (block_diag + cross_term).reshape(MK, MK) - 2 * ridge * np.eye(MK)

            step = np.linalg.solve(H, grad.reshape(MK)).reshape(M, K)

            # Backtracking, same as EMEnsembleOptimizer._fit_beta_m -- guards
            # against the non-quadratic tail of the objective far from the
            # optimum; near convergence this exits after one try.
            new_theta, new_Q = theta - step, -np.inf
            for _ in range(10):
                new_Q = objective(new_theta)
                if new_Q >= Q:
                    break
                step /= 2
                new_theta = theta - step

            converged = abs(new_Q - Q) < tol * (abs(Q) + 1e-12)
            theta, Q = new_theta, new_Q
            if verbose:
                print(f"Newton iter {it}: weighted log-lik = {Q:.4f}")
            if converged:
                if verbose:
                    print(f"Converged at iter {it} (delta log-lik = {new_Q - Q:.2e})")
                break

        self.beta = theta
        if verbose:
            print(f"Final weighted log-lik = {Q:.4f}")
        return self

#This code comes from https://github.com/itakurah/Focal-loss-PyTorch/blob/main/focal_loss.py
class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None, reduction='mean', task_type='binary', num_classes=None):
        """
        Unified Focal Loss class for binary, multi-class, and multi-label classification tasks.
        :param gamma: Focusing parameter, controls the strength of the modulating factor (1 - p_t)^gamma
        :param alpha: Balancing factor, can be a scalar or a tensor for class-wise weights. If None, no class balancing is used.
        :param reduction: Specifies the reduction method: 'none' | 'mean' | 'sum'
        :param task_type: Specifies the type of task: 'binary', 'multi-class', or 'multi-label'
        :param num_classes: Number of classes (only required for multi-class classification)
        """
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = 1e-10
        self.reduction = reduction
        self.task_type = task_type
        self.num_classes = num_classes

        # Handle alpha for class balancing in multi-class tasks
        if task_type == 'multi-class' and alpha is not None and isinstance(alpha, (list, torch.Tensor)):
            assert num_classes is not None, "num_classes must be specified for multi-class classification"
            if isinstance(alpha, list):
                self.alpha = torch.Tensor(alpha)
            else:
                self.alpha = alpha

    def forward(self, inputs, targets):
        """
        Forward pass to compute the Focal Loss based on the specified task type.
        :param inputs: Predictions (logits) from the model.
                       Shape:
                         - binary/multi-label: (batch_size, num_classes)
                         - multi-class: (batch_size, num_classes)
        :param targets: Ground truth labels.
                        Shape:
                         - binary: (batch_size,)
                         - multi-label: (batch_size, num_classes)
                         - multi-class: (batch_size,)
        """
        if self.task_type == 'binary':
            return self.binary_focal_loss(inputs, targets)
        elif self.task_type == 'multi-class':
            return self.multi_class_focal_loss(inputs, targets)
        elif self.task_type == 'multi-label':
            return self.multi_label_focal_loss(inputs, targets)
        else:
            raise ValueError(
                f"Unsupported task_type '{self.task_type}'. Use 'binary', 'multi-class', or 'multi-label'.")

    def binary_focal_loss(self, inputs, targets):
        """ Focal loss for binary classification. """
        probs = torch.sigmoid(inputs)
        targets = targets.float()

        # Compute binary cross entropy
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')

        # Compute focal weight
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma

        # Apply alpha if provided
        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            bce_loss = alpha_t * bce_loss

        # Apply focal loss weighting
        loss = focal_weight * bce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss

    def multi_class_focal_loss(self, inputs, targets):
        """ Focal loss for multi-class classification. """
        if self.alpha is not None:
            alpha = self.alpha.to(inputs.device)

        # Convert logits to probabilities with softmax
        #In our case, we are reweighting the probabilities across models, so the resuling probability already lives in the simplex
        probs = inputs #F.softmax(inputs, dim=1)

        # One-hot encode the targets
        targets = targets.argmax(dim=1)
        targets_one_hot = F.one_hot(targets, num_classes=self.num_classes).float()

        # Compute cross-entropy for each class
        ce_loss = -targets_one_hot * torch.log(probs+self.epsilon)

        # Compute focal weight
        p_t = torch.sum(probs * targets_one_hot, dim=1)  # p_t for each sample
        focal_weight = (1 - p_t) ** self.gamma

        # Apply alpha if provided (per-class weighting)
        if self.alpha is not None:
            alpha_t = alpha.gather(0, targets)
            ce_loss = alpha_t.unsqueeze(1) * ce_loss

        # Apply focal loss weight
        loss = focal_weight.unsqueeze(1) * ce_loss
        if torch.isnan(loss).any() or torch.isnan(inputs).any() or torch.isnan(targets).any():
            raise ValueError("NaN detected in loss or model outputs/targets")

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss

    def multi_label_focal_loss(self, inputs, targets):
        """ Focal loss for multi-label classification. """
        probs = torch.sigmoid(inputs)

        # Compute binary cross entropy
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')

        # Compute focal weight
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma

        # Apply alpha if provided
        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            bce_loss = alpha_t * bce_loss

        # Apply focal loss weight
        loss = focal_weight * bce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss

def create_loss_fn(weight=None, use_focal=True, gamma=2.0, task_type='multi-class', reduction='mean'):
    """
    Creates a loss function for classification, supporting both CrossEntropyLoss and FocalLoss.

    :param weight: Tensor of per-class weights (used as `alpha` in FocalLoss or `weight` in CrossEntropyLoss)
    :param use_focal: Whether to use Focal Loss
    :param gamma: Focusing parameter for Focal Loss
    :param task_type: 'binary', 'multi-class', or 'multi-label'
    :param num_classes: Required for multi-class focal loss
    :param reduction: 'mean', 'sum', or 'none'
    """
    if use_focal:
        return FocalLoss(
            gamma=gamma,
            alpha=weight,
            reduction=reduction,
            task_type=task_type,
            num_classes=len(weight)
        )
    else:
        return nn.CrossEntropyLoss(weight=weight, reduction=reduction)