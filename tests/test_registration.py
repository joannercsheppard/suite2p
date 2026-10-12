import numpy as np
import pytest
import torch
from suite2p.registration import bidiphase, rigid
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from suite2p.registration.nonrigid import transform_data
from suite2p.registration.utils import temporal_smooth


def test_positive_bidiphase_shift_shifts_every_other_line():
    orig = np.array([
        [[1, 2, 3, 4, 5, 6, 7],
         [1, 2, 3, 4, 5, 6, 7],
         [1, 2, 3, 4, 5, 6, 7],
         [1, 2, 3, 4, 5, 6, 7],
         [1, 2, 3, 4, 5, 6, 7]]
    ])
    expected = np.array([
        [[1, 2, 3, 4, 5, 6, 7],
         [1, 2, 1, 2, 3, 4, 5],
         [1, 2, 3, 4, 5, 6, 7],
         [1, 2, 1, 2, 3, 4, 5],
         [1, 2, 3, 4, 5, 6, 7]]
    ])

    shifted = orig.copy()
    bidiphase.shift(shifted, 2)
    assert np.allclose(shifted, expected)


def test_negative_bidiphase_shift_shifts_every_other_line():
    orig = np.array([
        [[1, 2, 3, 4, 5, 6, 7],
         [1, 2, 3, 4, 5, 6, 7],
         [1, 2, 3, 4, 5, 6, 7],
         [1, 2, 3, 4, 5, 6, 7],
         [1, 2, 3, 4, 5, 6, 7]]
    ])
    expected = np.array([
        [[1, 2, 3, 4, 5, 6, 7],
         [3, 4, 5, 6, 7, 6, 7],
         [1, 2, 3, 4, 5, 6, 7],
         [3, 4, 5, 6, 7, 6, 7],
         [1, 2, 3, 4, 5, 6, 7]]
    ])

    shifted = orig.copy()
    bidiphase.shift(shifted, -2)
    assert np.allclose(shifted, expected)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available")
def test_transform_data_mps_cpu_consistency():
    """Test that MPS and CPU code paths in transform_data produce similar results."""
    from suite2p.registration.nonrigid import make_blocks

    np.random.seed(42)
    torch.manual_seed(42)
    Ly, Lx, n_frames = 128, 128, 2
    yblock, xblock, nblocks, *_ = make_blocks(Ly, Lx, (32, 32))
    data_np = np.random.rand(n_frames, Ly, Lx).astype(np.float32) * 100
    ymax1 = torch.randn(nblocks[0] * nblocks[1], n_frames) * 2
    xmax1 = torch.randn(nblocks[0] * nblocks[1], n_frames) * 2

    result_cpu = transform_data(
        torch.from_numpy(data_np), nblocks, xblock, yblock, ymax1.clone(), xmax1.clone()
    )
    result_mps = transform_data(
        torch.from_numpy(data_np).to("mps"), nblocks, xblock, yblock,
        ymax1.clone().to("mps"), xmax1.clone().to("mps")
    )

    cpu_np = result_cpu.numpy().astype(np.float32)
    mps_np = result_mps.cpu().numpy().astype(np.float32)
    correlation = np.corrcoef(cpu_np.flatten(), mps_np.flatten())[0, 1]
    max_diff = np.abs(cpu_np - mps_np).max()

    assert correlation > 0.99, f"Correlation: {correlation}"
    assert max_diff < 2, f"Max diff: {max_diff}"


DEVICES = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else []) \
                  + (["cuda"] if torch.cuda.is_available() else [])


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("nimg, sigma", [(50, 1.0), (50, 2.5), (200, 0.3), (3, 2.0), (1, 1.0)])
def test_temporal_smooth_torch_matches_numpy(device, nimg, sigma):
    """Torch path of temporal_smooth matches scipy, incl. movies shorter than the kernel."""
    data_np = np.random.default_rng(0).standard_normal((nimg, 7, 9)).astype(np.float32)

    expected = temporal_smooth(data_np, sigma)
    result = temporal_smooth(torch.from_numpy(data_np).to(device), sigma)

    assert isinstance(result, torch.Tensor)
    assert result.device.type == device
    assert result.dtype == torch.float32
    np.testing.assert_allclose(result.cpu().numpy(), expected, atol=1e-5)


@pytest.mark.parametrize("device", DEVICES)
def test_phasecorr_temporal_smoothing_recovers_shift_in_noisy_frames(device):
    """With smooth_sigma_time > 0, phasecorr runs temporal_smooth on the torch correlation
    maps and recovers a known shift from frames too noisy to register one at a time."""
    rng = np.random.default_rng(0)
    ref = gaussian_filter(rng.standard_normal((128, 128)), 2).astype(np.float32)
    shift = (3, -5)
    noise = 8 * ref.std()
    frames = np.stack([np.roll(ref, shift, axis=(0, 1)) + noise * rng.standard_normal(ref.shape)
                       for _ in range(50)]).astype(np.float32)

    maskMul, maskOffset, cfRefImg = rigid.compute_masks_ref_smooth_fft(
        torch.from_numpy(ref), maskSlope=10, smooth_sigma=1.15)
    args = (torch.from_numpy(frames).to(device), cfRefImg.to(torch.complex64).to(device),
            maskMul.float().to(device), maskOffset.float().to(device), 0.1)

    def fraction_correct(smooth_sigma_time):
        ymax, xmax, _, _ = rigid.phasecorr(*args, smooth_sigma_time)
        return ((ymax.cpu().numpy() == shift[0]) & (xmax.cpu().numpy() == shift[1])).mean()

    # without smoothing some frames are misregistered, so the test is sensitive to smoothing
    assert fraction_correct(0) < 0.9
    assert fraction_correct(2.0) == 1.0
