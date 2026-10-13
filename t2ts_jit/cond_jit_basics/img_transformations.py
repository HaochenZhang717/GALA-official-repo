from abc import ABC, abstractmethod
import torch




class TsImgEmbedder(ABC):
    """
    Abstract class for transforming time series to images and vice versa
    """

    def __init__(self, device, seq_len):
        self.device = device
        self.seq_len = seq_len

    @abstractmethod
    def ts_to_img(self, signal):
        """

        Args:
            signal: given time series

        Returns:
            image representation of the signal

        """
        pass

    @abstractmethod
    def img_to_ts(self, img):
        """

        Args:
            img: given generated image

        Returns:
            time series representation of the generated image
        """
        pass


class DelayEmbedder(TsImgEmbedder):
    """
    Delay embedding transformation
    """

    def __init__(self, device, seq_len, delay, embedding):
        super().__init__(device, seq_len)
        self.delay = delay
        self.embedding = embedding
        self.img_shape = None

    def pad_to_square(self, x, mask=0):
        """
        Pads the input tensor x to make it square along the last two dimensions.
        """
        _, _, cols, rows = x.shape
        max_side = max(cols, rows)
        padding = (
            0, max_side - rows, 0, max_side - cols)  # Padding format: (pad_left, pad_right, pad_top, pad_bottom)

        # Padding the last two dimensions to make them square
        x_padded = torch.nn.functional.pad(x, padding, mode='constant', value=mask)
        return x_padded

    def unpad(self, x, original_shape):
        """
        Removes the padding from the tensor x to get back to its original shape.
        """
        _, _, original_cols, original_rows = original_shape
        return x[:, :, :original_cols, :original_rows]

    def ts_to_img(self, signal, pad=True, mask=0, return_pad_mask=False):

        batch, length, features = signal.shape
        #  if our sequences are of different lengths, this can happen with physionet and climate datasets
        if self.seq_len != length:
            self.seq_len = length

        x_image = torch.zeros(
            (batch, features, self.embedding, self.embedding),
            dtype=signal.dtype,
            device=signal.device,
        )
        pad_mask = torch.zeros_like(x_image)
        i = 0
        while (i * self.delay + self.embedding) <= self.seq_len:
            start = i * self.delay
            end = start + self.embedding
            x_image[:, :, :, i] = signal[:, start:end].permute(0, 2, 1)
            pad_mask[:, :, :, i] = 1
            i += 1

        ### SPECIAL CASE
        if i < self.embedding and i * self.delay != self.seq_len and i * self.delay + self.embedding > self.seq_len:
            start = i * self.delay
            end = signal[:, start:].permute(0, 2, 1).shape[-1]
            # end = start + (self.embedding - 1) - missing_vals
            x_image[:, :, :end, i] = signal[:, start:].permute(0, 2, 1)
            pad_mask[:, :, :end, i] = 1
            i += 1

        # cache the shape of the image before padding
        self.img_shape = (batch, features, self.embedding, i)
        x_image = x_image.to(self.device)[:, :, :, :i]
        pad_mask = pad_mask.to(self.device)[:, :, :, :i]

        if pad:
            x_image = self.pad_to_square(x_image, mask)
            pad_mask = self.pad_to_square(pad_mask, 0)

        if return_pad_mask:
            return x_image, pad_mask
        return x_image

    def _compute_n_cols(self) -> int:
        """Replay the column-count logic from `ts_to_img` so we can recover
        the pre-padding image shape without having cached it."""
        n_cols = 0
        while (n_cols * self.delay + self.embedding) <= self.seq_len:
            n_cols += 1
        # Special-case partial column: include leftover tail only if there's
        # still room in the image (n_cols < embedding) and the tail isn't
        # already aligned with the end of the signal.
        if (
            n_cols < self.embedding
            and n_cols * self.delay != self.seq_len
            and n_cols * self.delay + self.embedding > self.seq_len
        ):
            n_cols += 1
        return n_cols

    def img_to_ts(self, img):
        if self.img_shape is None:
            # No cached shape -> assume `img` came from an embedding with the
            # current (seq_len, delay, embedding) parameters and is padded to
            # square. Reconstruct the non-square shape `ts_to_img` would have
            # produced, taking batch/channel sizes from the actual tensor.
            batch, channels = img.shape[0], img.shape[1]
            self.img_shape = (batch, channels, self.embedding, self._compute_n_cols())

        img_non_square = self.unpad(img, self.img_shape)

        batch, channels, rows, cols = img_non_square.shape

        reconstructed_x_time_series = torch.zeros(
            (batch, channels, self.seq_len),
            dtype=img.dtype,
            device=img.device,
        )

        for i in range(cols - 1):
            start = i * self.delay
            end = start + self.embedding
            reconstructed_x_time_series[:, :, start:end] = img_non_square[:, :, :, i]

        ### SPECIAL CASE
        start = (cols - 1) * self.delay
        end = reconstructed_x_time_series[:, :, start:].shape[-1]
        reconstructed_x_time_series[:, :, start:] = img_non_square[:, :, :end, cols - 1]
        reconstructed_x_time_series = reconstructed_x_time_series.permute(0, 2, 1)

        return reconstructed_x_time_series.to(self.device)


def _test_delay_embedder():
    """Round-trip sanity checks for DelayEmbedder.

    The embedding fully covers the signal (so round-trip is lossless) when:
      - `(seq_len - embedding) // delay + 1 <= embedding`        (columns fit)
      - the loop + special-case together cover every index in [0, seq_len).

    We exercise three regimes:
      (1) exact fit, no special-case column,
      (2) partial tail that triggers the special-case column,
      (3) `img_to_ts` called on a fresh embedder (cached `img_shape == None`),
          which must reconstruct the shape from (seq_len, delay, embedding).
    """
    torch.manual_seed(0)
    device = "cpu"

    def _round_trip(seq_len, delay, embedding, batch=2, features=3, use_cached_shape=True):
        signal = torch.randn(batch, seq_len, features, device=device)
        emb = DelayEmbedder(device=device, seq_len=seq_len, delay=delay, embedding=embedding)
        img = emb.ts_to_img(signal)
        if not use_cached_shape:
            emb.img_shape = None  # force the `???` fallback path
        recon = emb.img_to_ts(img)
        assert recon.shape == signal.shape, f"shape mismatch: {recon.shape} vs {signal.shape}"
        return signal, recon

    # (1) Exact fit: seq_len=16, delay=2, embedding=8 -> 5 columns, no special case.
    #     (16 - 8) // 2 + 1 = 5 <= 8. Last window starts at 8, ends at 16 = seq_len.
    sig, rec = _round_trip(seq_len=16, delay=2, embedding=8)
    assert torch.allclose(sig, rec), "exact-fit round-trip failed"

    # (2) Partial tail triggers special-case column:
    #     seq_len=10, delay=2, embedding=5 -> i=3 full cols, then one partial col.
    sig, rec = _round_trip(seq_len=10, delay=2, embedding=5)
    assert torch.allclose(sig, rec), "special-case round-trip failed"

    # (3) Same params as (2) but force the fallback that recomputes img_shape.
    sig, rec = _round_trip(seq_len=10, delay=2, embedding=5, use_cached_shape=False)
    assert torch.allclose(sig, rec), "fallback (img_shape=None) round-trip failed"

    # _compute_n_cols agrees with what ts_to_img caches.
    emb = DelayEmbedder(device=device, seq_len=10, delay=2, embedding=5)
    emb.ts_to_img(torch.randn(1, 10, 1))
    assert emb.img_shape[-1] == emb._compute_n_cols(), "_compute_n_cols disagrees with cached img_shape"

    print("DelayEmbedder round-trip tests passed.")


if __name__ == "__main__":
    _test_delay_embedder()
