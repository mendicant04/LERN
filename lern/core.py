"""Causal local-margin rank adaptation using only previously arrived images.

The supplied text prototypes remain fixed throughout the stream.
Odd arrivals enter a reference bank; even arrivals enter a calibration bank.
At powers of two arrivals, the reference scorer is refreshed and every past
calibration image is rescored with that same scorer. Between refreshes, new
calibration images extend its reference distribution. Emitted predictions
never change. No method input contains labels or the eventual stream length.
"""
import hashlib

import numpy as np
import torch


def unit(x):
    return x / torch.linalg.vector_norm(x, dim=-1, keepdim=True).clamp_min(1e-12)


class _Bank:
    def __init__(self, dimensions, classes, device):
        self.dimensions = dimensions
        self.classes = classes
        self.device = device
        self.n = 0
        self.capacity = 0
        self.h = self.margins = self.ids = None

    def append(self, h, margins, ids):
        required = self.n + len(h)
        if required > self.capacity:
            capacity = max(8, 1 << (required - 1).bit_length())
            hh = torch.empty((capacity, self.dimensions), dtype=torch.float64,
                             device=self.device)
            aa = torch.empty((capacity, self.classes), dtype=torch.float64,
                             device=self.device)
            ii = torch.empty(capacity, dtype=torch.int64, device=self.device)
            if self.n:
                hh[:self.n] = self.h[:self.n]
                aa[:self.n] = self.margins[:self.n]
                ii[:self.n] = self.ids[:self.n]
            self.h, self.margins, self.ids = hh, aa, ii
            self.capacity = capacity
        if len(h):
            self.h[self.n:required] = h
            self.margins[self.n:required] = margins
            self.ids[self.n:required] = ids
        self.n = required


class OnlineLocalStatistic:
    """Observe arrivals and emit one irrevocable prediction per image.

    Batching is a computational convenience. Within each batch, calibration
    counts are masked by arrival position and the active reference pool is
    already in the past. observe(..., chunk=1) is the single-image interface.
    """

    def __init__(self, text, dataset, order_seed, device="cpu"):
        self.device = torch.device(device)
        self.text = unit(torch.as_tensor(text, dtype=torch.float64,
                                         device=self.device))
        if self.text.ndim != 2 or len(self.text) < 2:
            raise ValueError("Use a C x D text head with at least two classes.")
        if not torch.isfinite(self.text).all():
            raise ValueError("Text features must be finite.")
        self.classes = len(self.text)
        self.dataset = str(dataset)
        self.order_seed = int(order_seed)
        self.n = 0
        self.reference = self.calibration = None
        self.active_h = self.active_margins = None
        self.sorted_calibration = None
        self.calibration_scores = None
        self.reference_prefix_end = 0
        self.k = 0
        self.ready = False
        self.seen_ids = set()
        self.refreshes = []

    def _local(self, h, chunk=256):
        parts = []
        for start in range(0, len(h), chunk):
            similarity = h[start:start + chunk] @ self.active_h.T
            values, near = torch.topk(similarity, self.k, dim=1,
                                     largest=True, sorted=True)
            cutoff = values[:, -1:]
            strict = (similarity > cutoff).sum(1)
            equal = (similarity == cutoff).sum(1)
            boundary_ties = equal > (self.k - strict)
            if boundary_ties.any():
                # Active references are sorted by sample ID.
                tied = torch.nonzero(boundary_ties).flatten()
                near[tied] = torch.argsort(similarity[tied], dim=1,
                                          descending=True, stable=True)[:, :self.k]
            # Canonical summation makes identical neighbor sets exactly equal
            # even when similarity order within a set differs.
            near = torch.sort(near, dim=1).values
            parts.append(self.active_margins[near].mean(1))
        if not parts:
            return torch.empty((0, self.classes), dtype=torch.float64,
                               device=self.device)
        return torch.cat(parts)

    def _refresh(self):
        r, c = self.reference, self.calibration
        self.reference_prefix_end = self.n
        self.k = int(np.ceil(np.sqrt(r.n)))
        order = torch.argsort(r.ids[:r.n], stable=True)
        self.active_h = r.h[:r.n][order].clone()
        self.active_margins = r.margins[:r.n][order].clone()
        # If k equals the whole reference pool, every image receives the same
        # local statistic. There is no image-specific rank evidence yet.
        self.ready = c.n > 0 and self.k < r.n
        if self.ready:
            self.calibration_scores = self._local(c.h[:c.n])
            self.sorted_calibration = torch.sort(
                self.calibration_scores.T.contiguous(), dim=1).values.contiguous()
        self.refreshes.append(dict(prefix=self.n, references=r.n,
                                   calibration=c.n, k=self.k,
                                   active=self.ready))

    def _choose(self, scores, ids):
        maximum = scores == scores.max(1, keepdim=True).values
        tied = maximum.sum(1) > 1
        prediction = scores.argmax(1)
        for row in torch.nonzero(tied).flatten().cpu().tolist():
            candidates = torch.nonzero(maximum[row]).flatten().cpu().tolist()
            def key(cls):
                message = (f"{self.dataset}:{self.order_seed}:{int(ids[row])}:"
                           f"class:{cls}:online-statistic-tie")
                return int.from_bytes(hashlib.sha256(message.encode()).digest()[:8],
                                      "big")
            prediction[row] = max(candidates, key=key)
        return prediction, tied

    @torch.no_grad()
    def observe(self, images, preprojection, ids, chunk=256):
        if chunk < 1:
            raise ValueError("chunk must be positive.")
        x = torch.as_tensor(images, dtype=torch.float64, device=self.device)
        h = torch.as_tensor(preprojection, dtype=torch.float64, device=self.device)
        ids = np.asarray(ids, dtype=np.int64)
        if (x.ndim != 2 or h.ndim != 2 or len(x) != len(h)
                or ids.shape != (len(x),) or x.shape[1] != self.text.shape[1]):
            raise ValueError("Feature arrays and stable IDs must have matching rows.")
        if self.reference is None:
            self.reference = _Bank(h.shape[1], self.classes, self.device)
            self.calibration = _Bank(h.shape[1], self.classes, self.device)
        if h.shape[1] != self.reference.dimensions:
            raise ValueError("Preprojection dimensionality changed.")
        if not torch.isfinite(x).all() or not torch.isfinite(h).all():
            raise ValueError("Image features must be finite.")
        if len(np.unique(ids)) != len(ids) or any(int(i) in self.seen_ids for i in ids):
            raise ValueError("An image ID cannot enter the same stream twice.")
        x, h = unit(x), unit(h)
        all_results = []
        start = 0
        while start < len(x):
            next_refresh = 1 << self.n.bit_length()
            amount = min(chunk, len(x) - start, next_refresh - self.n)
            end = start + amount
            xx, hh, ii = x[start:end], h[start:end], ids[start:end]
            cosine = xx @ self.text.T
            frozen = cosine.argmax(1)
            best = cosine.topk(2, dim=1).values
            competitor = best[:, :1].expand_as(cosine).clone()
            competitor.scatter_(1, frozen[:, None], best[:, 1:2])
            margins = cosine - competitor
            positions = torch.arange(self.n + 1, self.n + amount + 1,
                                     device=self.device, dtype=torch.int64)
            is_calibration = positions.remainder(2) == 0
            calibration_before = self.calibration.n + (
                is_calibration.to(torch.int64).cumsum(0)
                - is_calibration.to(torch.int64))
            active_count = 0 if self.active_h is None else len(self.active_h)
            if self.ready:
                g = self._local(hh, chunk=chunk)
                query = g.T.contiguous()
                left = torch.searchsorted(self.sorted_calibration, query,
                                          right=False).T
                right = torch.searchsorted(self.sorted_calibration, query,
                                           right=True).T
                # Row i may use only calibration arrivals strictly before i.
                previous = (torch.arange(amount, device=self.device)[None, :]
                            < torch.arange(amount, device=self.device)[:, None])
                previous &= is_calibration[None, :]
                left += ((g[None, :, :] < g[:, None, :])
                         & previous[:, :, None]).sum(1)
                right += ((g[None, :, :] <= g[:, None, :])
                          & previous[:, :, None]).sum(1)
                numerators = left + right + 1
                denominators = 2 * (calibration_before + 1)
                prediction, tied = self._choose(numerators, ii)
                local_prediction, _ = self._choose(g, ii)
            else:
                g = torch.full_like(cosine, torch.nan)
                numerators = torch.zeros_like(cosine, dtype=torch.int64)
                denominators = torch.zeros(amount, dtype=torch.int64,
                                           device=self.device)
                prediction = local_prediction = frozen
                tied = torch.zeros(amount, dtype=torch.bool, device=self.device)
            result = dict(
                indices=ii.copy(),
                stream_positions=positions.cpu().numpy(),
                frozen=frozen.cpu().numpy(),
                online_local_statistic=prediction.cpu().numpy(),
                online_local_margin=local_prediction.cpu().numpy(),
                rank_numerators=numerators.cpu().numpy(),
                rank_denominators=denominators.cpu().numpy(),
                local_statistic=g.cpu().numpy(),
                maximum_tie=tied.cpu().numpy(),
                is_adapted=np.full(amount, self.ready, dtype=bool),
                reference_count=np.full(amount, active_count, dtype=np.int64),
                calibration_count=calibration_before.cpu().numpy(),
                reference_prefix_end=np.full(amount, self.reference_prefix_end,
                                             dtype=np.int64),
            )
            all_results.append(result)
            id_tensor = torch.as_tensor(ii, device=self.device, dtype=torch.int64)
            self.reference.append(hh[~is_calibration], margins[~is_calibration],
                                  id_tensor[~is_calibration])
            self.calibration.append(hh[is_calibration], margins[is_calibration],
                                    id_tensor[is_calibration])
            if self.ready and is_calibration.any():
                self.calibration_scores = torch.cat(
                    [self.calibration_scores, g[is_calibration]], dim=0)
                self.sorted_calibration = torch.sort(
                    self.calibration_scores.T.contiguous(), dim=1).values.contiguous()
            self.seen_ids.update(int(i) for i in ii)
            self.n += amount
            start = end
            if self.n == next_refresh:
                self._refresh()
        if not all_results:
            raise ValueError("Observe at least one image.")
        return {key: np.concatenate([part[key] for part in all_results])
                for key in all_results[0]}
