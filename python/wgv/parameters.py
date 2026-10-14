"""Typed, reproducible parameter sampling with explicit rejection."""
import math
import numpy as np


def draw(spec, rng, *, integer=False, minimum=None, maximum=None):
    """Sample a scalar; Gaussian integer draws round nearest, half ties away from zero.

    Normal/gaussian integer bounds condition the rounded draws by rejection.
    Truncated-normal bounds condition the continuous draw before rounding.
    Other laws and fixed values retain strict integer validation.
    """
    if isinstance(spec, (list, tuple)):
        if len(spec) != 2:
            raise ValueError("a scalar range requires exactly [minimum, maximum]")
        spec = {"distribution": "uniform_int" if integer else "uniform",
                "min": spec[0], "max": spec[1]}
    if not isinstance(spec, dict):
        value = spec
    else:
        name = spec.get("distribution", ("uniform_int" if integer else "uniform")
                        if "min" in spec and "max" in spec and "value" not in spec else "fixed")
        if name == "fixed":
            value = spec["value"]
        elif name == "uniform_int":
            low, high = spec["min"], spec["max"]
            if int(low) != low or int(high) != high or low > high:
                raise ValueError("uniform_int requires ordered integer bounds")
            value = int(rng.integers(low, high + 1))
        elif name == "uniform":
            if spec["min"] > spec["max"]:
                raise ValueError("uniform bounds are reversed")
            value = float(rng.uniform(spec["min"], spec["max"]))
        elif name in ("normal", "gaussian", "truncated_normal"):
            mean, std = spec["mean"], spec["std"]
            if not math.isfinite(mean) or not math.isfinite(std) or std < 0:
                raise ValueError("normal requires a finite mean and nonnegative finite standard deviation")
            truncated = name == "truncated_normal"
            bounded_integer = integer and ("min" in spec or "max" in spec)
            low = spec["min"] if truncated else spec.get("min", -math.inf)
            high = spec["max"] if truncated else spec.get("max", math.inf)
            if truncated or bounded_integer:
                for key in ("min", "max"):
                    if key in spec and (not math.isfinite(spec[key]) or
                                        (integer and int(spec[key]) != spec[key])):
                        raise ValueError("Gaussian bounds must be finite; integer parameters require integer bounds")
                if low > high:
                    raise ValueError("Gaussian bounds are reversed")
            for _ in range(10000):
                value = float(rng.normal(mean, std))
                if not math.isfinite(value):
                    raise ValueError("normal produced a nonfinite value")
                if truncated and not low <= value <= high:
                    continue
                if integer:
                    value = math.floor(value + .5) if value >= 0 else math.ceil(value - .5)
                if bounded_integer and not low <= value <= high:
                    continue
                break
            else:
                raise ValueError("Gaussian failed to produce a feasible value")
        elif name == "lognormal":
            if spec["sigma"] < 0:
                raise ValueError("lognormal sigma must be nonnegative")
            value = float(rng.lognormal(spec.get("mean", 0), spec["sigma"]))
        elif name == "exponential":
            if spec["scale"] <= 0:
                raise ValueError("exponential scale must be positive")
            value = float(rng.exponential(spec["scale"]))
        elif name == "poisson":
            if spec["lambda"] < 0:
                raise ValueError("Poisson lambda must be nonnegative")
            value = int(rng.poisson(spec["lambda"]))
        elif name == "beta":
            low, high = spec.get("min", 0), spec.get("max", 1)
            if spec["alpha"] <= 0 or spec["beta"] <= 0 or low > high:
                raise ValueError("beta requires positive shape parameters and ordered bounds")
            value = float(low + (high-low)*rng.beta(spec["alpha"], spec["beta"]))
        elif name == "categorical":
            values, weights = spec["values"], spec.get("weights")
            if not values:
                raise ValueError("categorical requires values")
            if weights is not None:
                weights = np.asarray(weights, dtype=float)
                if len(weights) != len(values) or not np.all(np.isfinite(weights)) or np.any(weights < 0) or weights.sum() <= 0:
                    raise ValueError("invalid categorical weights")
                weights = weights / weights.sum()
            value = values[int(rng.choice(len(values), p=weights))]
        elif name == "mixture":
            parts = spec["components"]
            weights = np.asarray(spec.get("weights", np.ones(len(parts))), float)
            if not parts or len(weights) != len(parts) or not np.all(np.isfinite(weights)) or np.any(weights < 0) or weights.sum() <= 0:
                raise ValueError("invalid mixture")
            value = draw(parts[int(rng.choice(len(parts), p=weights/weights.sum()))], rng,
                         integer=integer, minimum=minimum, maximum=maximum)
        else:
            raise ValueError(f"unsupported distribution: {name}")
    if isinstance(value, str):
        if integer or minimum is not None or maximum is not None:
            raise ValueError("a numeric parameter cannot be categorical text")
        return value
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)) or not math.isfinite(value):
        raise ValueError(f"expected a finite scalar, got {value!r}")
    if integer and int(value) != value:
        raise ValueError("integer parameters require integer-valued draws or a Gaussian law with rounding")
    if minimum is not None and value < minimum:
        raise ValueError(f"value {value} is below minimum {minimum}")
    if maximum is not None and value > maximum:
        raise ValueError(f"value {value} exceeds maximum {maximum}")
    return int(value) if integer else float(value)


def draw_point(spec, rng, dimension):
    """Draw a whole location, including correlated Gaussian and mixture laws.

    Coordinate units are interpreted by the caller; this function never clips
    a draw to a world or substitutes a center-containment test.
    """
    if not isinstance(spec, dict):
        if not isinstance(spec, (list, tuple)) or len(spec) != dimension:
            raise ValueError("a point requires one scalar specification per axis")
        return np.asarray([draw(s, rng) for s in spec], float)
    kind = spec.get("type", spec.get("distribution", "independent"))
    if kind == "fixed":
        return draw_point(spec["value"], rng, dimension)
    if kind == "independent":
        axes = ("x", "y", "z")[:dimension]
        if any(axis not in spec for axis in axes):
            raise ValueError(f"independent placement requires axes {axes}")
        return np.asarray([draw(spec[axis], rng) for axis in axes])
    if kind == "uniform":
        low, high = np.asarray(spec["min"], float), np.asarray(spec["max"], float)
        if low.shape != (dimension,) or high.shape != (dimension,):
            raise ValueError("placement bounds must match the point dimension")
        return draw_point([[a, b] for a, b in zip(low, high)], rng, dimension)
    if kind in ("gaussian", "normal"):
        mean = np.asarray(spec["mean"], float)
        if mean.shape != (dimension,) or not np.all(np.isfinite(mean)):
            raise ValueError("Gaussian placement requires a finite mean per axis")
        if "covariance" in spec:
            covariance = np.asarray(spec["covariance"], float)
            if (covariance.shape != (dimension, dimension) or not np.all(np.isfinite(covariance))
                    or not np.allclose(covariance, covariance.T)
                    or np.linalg.eigvalsh(covariance).min() < -1e-12):
                raise ValueError("placement covariance must be symmetric positive semidefinite")
            return rng.multivariate_normal(mean, covariance, check_valid="raise")
        std = np.broadcast_to(np.asarray(spec["std"], float), (dimension,))
        if not np.all(np.isfinite(std)) or np.any(std < 0):
            raise ValueError("placement standard deviations must be finite and nonnegative")
        return rng.normal(mean, std)
    if kind == "mixture":
        components = spec["components"]
        index = draw({"distribution": "categorical", "values": list(range(len(components))),
                      "weights": spec.get("weights")}, rng, integer=True)
        return draw_point(components[index], rng, dimension)
    raise ValueError(f"unsupported placement distribution: {kind}")
